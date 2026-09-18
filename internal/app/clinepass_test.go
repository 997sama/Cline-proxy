package app

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync/atomic"
	"testing"

	"cline-go-proxy/internal/kit"
)

func installClinePassTestState(t *testing.T, cfg *ClinePassConfig) {
	t.Helper()
	oldCfg := getClinePassConfig()
	oldClient := kit.HTTPClient
	clinePassConfigMu.Lock()
	clinePassConfig = cloneClinePassConfig(cfg)
	clinePassConfigMu.Unlock()
	clinePassMetadataMu.Lock()
	oldMetadata := clinePassMetadata
	clinePassMetadata = make(map[string]*ClinePassModelMetadata)
	clinePassMetadataMu.Unlock()
	t.Cleanup(func() {
		clinePassConfigMu.Lock()
		clinePassConfig = oldCfg
		clinePassConfigMu.Unlock()
		clinePassMetadataMu.Lock()
		clinePassMetadata = oldMetadata
		clinePassMetadataMu.Unlock()
		kit.HTTPClient = oldClient
		_ = os.Remove(clinePassConfigPath)
	})
}

func testClinePassConfig(policy ClinePassModelPolicy) *ClinePassConfig {
	return &ClinePassConfig{
		Enabled:     true,
		BaseURL:     "http://127.0.0.1:1/chat/completions",
		AccountMode: "round_robin",
		Accounts: []ClinePassAccount{{
			Name:    "test",
			APIKey:  "sk-test-secret",
			Enabled: true,
		}},
		PerModel: map[string]ClinePassModelPolicy{
			"cline-pass/test": policy,
		},
	}
}

func decodeRequestBody(t *testing.T, r *http.Request) map[string]any {
	t.Helper()
	var body map[string]any
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		t.Fatalf("decode request body: %v", err)
	}
	return body
}

func TestApplyClinePassProviderPolicy(t *testing.T) {
	tests := []struct {
		name   string
		policy ClinePassModelPolicy
		assert func(t *testing.T, body map[string]any)
	}{
		{
			name:   "planner strict",
			policy: ClinePassModelPolicy{Pipeline: "planner", Mode: "strict", Providers: []string{"deepseek"}},
			assert: func(t *testing.T, body map[string]any) {
				options := body["providerOptions"].(map[string]any)
				gateway := options["gateway"].(map[string]any)
				if got := gateway["only"].([]string); len(got) != 1 || got[0] != "deepseek" {
					t.Fatalf("unexpected planner only: %#v", got)
				}
				if _, ok := body["provider"]; ok {
					t.Fatal("direct provider leaked into planner body")
				}
			},
		},
		{
			name:   "direct strict",
			policy: ClinePassModelPolicy{Pipeline: "direct", Mode: "strict", Providers: []string{"deepseek"}},
			assert: func(t *testing.T, body map[string]any) {
				provider := body["provider"].(map[string]any)
				if got := provider["only"].([]string); len(got) != 1 || got[0] != "deepseek" {
					t.Fatalf("unexpected direct only: %#v", got)
				}
				if _, ok := body["providerOptions"]; ok {
					t.Fatal("planner options leaked into direct body")
				}
			},
		},
		{
			name:   "planner preferred sorted",
			policy: ClinePassModelPolicy{Pipeline: "planner", Mode: "preferred", Providers: []string{"deepseek", "novita"}, Sort: "cost"},
			assert: func(t *testing.T, body map[string]any) {
				gateway := body["providerOptions"].(map[string]any)["gateway"].(map[string]any)
				got := gateway["order"].([]string)
				if len(got) != 2 || got[0] != "deepseek" || got[1] != "novita" || gateway["sort"] != "cost" {
					t.Fatalf("unexpected planner preferred: %#v", gateway)
				}
			},
		},
		{
			name:   "direct preferred",
			policy: ClinePassModelPolicy{Pipeline: "direct", Mode: "preferred", Providers: []string{"deepseek", "novita"}},
			assert: func(t *testing.T, body map[string]any) {
				provider := body["provider"].(map[string]any)
				got := provider["order"].([]string)
				if len(got) != 2 || got[0] != "deepseek" || got[1] != "novita" {
					t.Fatalf("unexpected direct preferred: %#v", provider)
				}
			},
		},
		{
			name:   "auto does not pin",
			policy: ClinePassModelPolicy{Pipeline: "planner", Mode: "auto", Providers: []string{"deepseek"}},
			assert: func(t *testing.T, body map[string]any) {
				if _, ok := body["provider"]; ok {
					t.Fatal("auto unexpectedly pinned direct provider")
				}
				if _, ok := body["providerOptions"]; ok {
					t.Fatal("auto unexpectedly pinned gateway provider")
				}
			},
		},
		{
			name:   "exclude filters known candidates",
			policy: ClinePassModelPolicy{Pipeline: "direct", Mode: "auto", Providers: []string{"deepseek", "novita"}, Exclude: []string{"novita"}},
			assert: func(t *testing.T, body map[string]any) {
				provider := body["provider"].(map[string]any)
				got := provider["only"].([]string)
				if len(got) != 1 || got[0] != "deepseek" {
					t.Fatalf("excluded provider was not removed: %#v", provider)
				}
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body := map[string]any{
				"model":           "cline-pass/test",
				"provider":        map[string]any{"only": []string{"novita"}},
				"providerOptions": map[string]any{"gateway": map[string]any{"only": []string{"novita"}}},
			}
			tt.assert(t, applyClinePassProviderPolicy(body, "cline-pass/test", tt.policy))
		})
	}
}

func TestClinePassClientRoutingCannotOverrideServerPolicy(t *testing.T) {
	cfg := testClinePassConfig(ClinePassModelPolicy{Pipeline: "direct", Mode: "strict", Providers: []string{"deepseek"}})
	installClinePassTestState(t, cfg)
	body := buildClinePassBody(map[string]any{
		"model":           "cline-pass/test",
		"messages":        []any{},
		"provider":        map[string]any{"only": []string{"novita"}},
		"providerOptions": map[string]any{"gateway": map[string]any{"only": []string{"novita"}}},
	}, false)
	final := applyClinePassProviderPolicy(body, "cline-pass/test", cfg.PerModel["cline-pass/test"])
	provider := final["provider"].(map[string]any)
	if got := provider["only"].([]string); len(got) != 1 || got[0] != "deepseek" {
		t.Fatalf("client routing bypassed server policy: %#v", provider)
	}
}

func TestClinePassPipelineAndMetadataDetection(t *testing.T) {
	planner := map[string]any{"model": "cline-pass/test", "provider_metadata": map[string]any{"gateway": map[string]any{"routing": map[string]any{"canonicalSlug": "deepseek-v4.1-flash", "finalProvider": "deepseek"}}}}
	obs := observeClinePassPayload("cline-pass/test", planner)
	if obs.Pipeline != "planner" || obs.ActualProvider != "deepseek" || obs.ActualModel != "deepseek-v4.1-flash" {
		t.Fatalf("planner observation mismatch: %#v", obs)
	}
	direct := map[string]any{"model": "deepseek-v4.1-flash", "provider": "deepseek"}
	obs = observeClinePassPayload("cline-pass/test", direct)
	if obs.Pipeline != "direct" || obs.ActualProvider != "deepseek" || obs.ActualModel != "deepseek-v4.1-flash" {
		t.Fatalf("direct observation mismatch: %#v", obs)
	}
	unknown := observeClinePassPayload("cline-pass/test", map[string]any{"choices": []any{}})
	if unknown.Pipeline != "" || unknown.ActualProvider != "" || unknown.ActualModel != "" {
		t.Fatalf("unknown observation should be empty: %#v", unknown)
	}
}

func TestClinePassPreferredFallbackAfter429(t *testing.T) {
	cfg := testClinePassConfig(ClinePassModelPolicy{Pipeline: "planner", Mode: "preferred", Providers: []string{"deepseek", "novita"}})
	installClinePassTestState(t, cfg)
	var calls int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		call := atomic.AddInt32(&calls, 1)
		body := decodeRequestBody(t, r)
		if call == 1 {
			options, ok := body["providerOptions"].(map[string]any)
			if !ok {
				t.Errorf("first request missing providerOptions: %#v", body)
				w.WriteHeader(http.StatusInternalServerError)
				return
			}
			gateway, ok := options["gateway"].(map[string]any)
			if !ok {
				t.Errorf("first request missing gateway options: %#v", body)
				w.WriteHeader(http.StatusInternalServerError)
				return
			}
			order, ok := gateway["order"].([]any)
			if !ok {
				t.Errorf("first request missing order: %#v", gateway)
				w.WriteHeader(http.StatusInternalServerError)
				return
			}
			if len(order) != 2 || order[0] != "deepseek" || order[1] != "novita" {
				t.Errorf("first request did not use gateway order: %#v", gateway)
			}
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = w.Write([]byte(`{"error":"temporarily rate limited"}`))
			return
		}
		providerOptions := body["providerOptions"].(map[string]any)
		provider := providerOptions["gateway"].(map[string]any)
		only := provider["only"].([]any)
		if len(only) != 1 || only[0] != "novita" {
			t.Errorf("fallback request did not pin novita: %#v", provider)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"model":"deepseek-v4.1-flash","provider":"novita","choices":[]}`))
	}))
	defer server.Close()
	cfg.BaseURL = server.URL
	clinePassConfigMu.Lock()
	clinePassConfig.BaseURL = cfg.BaseURL
	clinePassConfigMu.Unlock()
	kit.HTTPClient = server.Client()
	resp, meta, err := callClinePassAPI(map[string]any{"model": "cline-pass/test", "messages": []any{}}, false)
	if err != nil {
		t.Fatalf("fallback returned error: %v", err)
	}
	defer resp.Body.Close()
	if got := atomic.LoadInt32(&calls); got != 2 {
		t.Fatalf("expected two attempts, got %d", got)
	}
	if len(meta.Attempts) != 2 || meta.Attempts[0].Kind != "rate_limit" || meta.Attempts[1].Provider != "novita" {
		t.Fatalf("unexpected attempts: %#v", meta.Attempts)
	}
}

type roundTripperFunc func(*http.Request) (*http.Response, error)

func (f roundTripperFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func TestClinePassPreferredFallbackAfterNetworkError(t *testing.T) {
	cfg := testClinePassConfig(ClinePassModelPolicy{Pipeline: "direct", Mode: "preferred", Providers: []string{"deepseek", "novita"}})
	installClinePassTestState(t, cfg)
	var calls int32
	kit.HTTPClient = &http.Client{Transport: roundTripperFunc(func(r *http.Request) (*http.Response, error) {
		call := atomic.AddInt32(&calls, 1)
		if call == 1 {
			return nil, errors.New("simulated dial failure")
		}
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     make(http.Header),
			Body:       io.NopCloser(strings.NewReader(`{"model":"deepseek-v4.1-flash","provider":"novita","choices":[]}`)),
		}, nil
	})}
	resp, meta, err := callClinePassAPI(map[string]any{"model": "cline-pass/test", "messages": []any{}}, false)
	if err != nil {
		t.Fatalf("network fallback returned error: %v", err)
	}
	defer resp.Body.Close()
	if calls != 2 || len(meta.Attempts) != 2 || meta.Attempts[0].Kind != "network_error" || meta.Attempts[1].Provider != "novita" {
		t.Fatalf("unexpected network fallback attempts: calls=%d meta=%#v", calls, meta.Attempts)
	}
}

func TestClinePassProtocolEntrypoints(t *testing.T) {
	cfg := testClinePassConfig(ClinePassModelPolicy{Pipeline: "direct", Mode: "strict", Providers: []string{"deepseek"}})
	installClinePassTestState(t, cfg)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body := decodeRequestBody(t, r)
		provider := body["provider"].(map[string]any)
		if provider["only"].([]any)[0] != "deepseek" {
			t.Errorf("unexpected provider policy: %#v", provider)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"model":"deepseek-v4.1-flash","provider":"deepseek","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`))
	}))
	defer server.Close()
	clinePassConfigMu.Lock()
	clinePassConfig.BaseURL = server.URL
	clinePassConfigMu.Unlock()
	kit.HTTPClient = server.Client()

	chatReq := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"cline-pass/test","messages":[{"role":"user","content":"hi"}]}`))
	chatW := httptest.NewRecorder()
	handleClinePassChat(chatW, chatReq, map[string]any{"model": "cline-pass/test", "messages": []any{}}, false)
	if chatW.Code != http.StatusOK || !strings.Contains(chatW.Body.String(), `"content":"ok"`) {
		t.Fatalf("chat entrypoint failed: %d %s", chatW.Code, chatW.Body.String())
	}

	responsesReq := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{"model":"cline-pass/test","input":"hi"}`))
	responsesW := httptest.NewRecorder()
	handleResponses(responsesW, responsesReq)
	if responsesW.Code != http.StatusOK || !strings.Contains(responsesW.Body.String(), `"output_text":"ok"`) {
		t.Fatalf("responses entrypoint failed: %d %s", responsesW.Code, responsesW.Body.String())
	}

	anthropicReq := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(`{"model":"cline-pass/test","max_tokens":16,"messages":[{"role":"user","content":"hi"}]}`))
	anthropicW := httptest.NewRecorder()
	handleAnthropicMessages(anthropicW, anthropicReq)
	if anthropicW.Code != http.StatusOK || !strings.Contains(anthropicW.Body.String(), `"text":"ok"`) {
		t.Fatalf("anthropic entrypoint failed: %d %s", anthropicW.Code, anthropicW.Body.String())
	}
}

type flushRecorder struct {
	*httptest.ResponseRecorder
	flushes int
}

func (w *flushRecorder) Flush() {
	w.flushes++
	w.ResponseRecorder.Flush()
}

func TestClinePassSSEUsesFlusherAndDoesNotRetryAfterBodyStarts(t *testing.T) {
	cfg := testClinePassConfig(ClinePassModelPolicy{Pipeline: "planner", Mode: "preferred", Providers: []string{"deepseek", "novita"}})
	installClinePassTestState(t, cfg)
	var calls int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&calls, 1)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"model\":\"deepseek-v4.1-flash\",\"provider_metadata\":{\"gateway\":{\"routing\":{\"finalProvider\":\"deepseek\"}}},\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n"))
	}))
	defer server.Close()
	clinePassConfigMu.Lock()
	clinePassConfig.BaseURL = server.URL
	clinePassConfigMu.Unlock()
	kit.HTTPClient = server.Client()
	resp, meta, err := callClinePassAPI(map[string]any{"model": "cline-pass/test", "messages": []any{}, "stream": true}, true)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	w := &flushRecorder{ResponseRecorder: httptest.NewRecorder()}
	handleStreamResponseWithUsageMeta(w, resp, nil, func(payload map[string]any) {
		recordClinePassObservation(meta.Model, meta, payload)
	})
	if w.flushes == 0 {
		t.Fatal("streaming response did not flush")
	}
	if got := atomic.LoadInt32(&calls); got != 1 {
		t.Fatalf("stream started and unexpectedly retried: %d requests", got)
	}
	if meta.ActualProvider != "deepseek" {
		t.Fatalf("actual provider not observed from SSE: %#v", meta)
	}
}

func TestClinePassProviderDiscoveryParsingAndKeyMasking(t *testing.T) {
	providers := parseAvailableClinePassProviders(`{"error":{"message":"Available providers are: deepseek, novita, fireworks"}}`)
	if len(providers) != 3 || providers[0] != "deepseek" || providers[2] != "fireworks" {
		t.Fatalf("unexpected discovered providers: %#v", providers)
	}
	if got := maskClinePassKey("sk-abcd123456xyz"); got != "sk-abcd****xyz" {
		t.Fatalf("unexpected masked key: %s", got)
	}
	if strings.Contains(maskClinePassKey("sk-abcd123456xyz"), "123456") {
		t.Fatal("masked key leaked secret middle")
	}
}

func TestClinePassAdminDoesNotExposeAPIKey(t *testing.T) {
	cfg := testClinePassConfig(ClinePassModelPolicy{Pipeline: "planner", Mode: "strict", Providers: []string{"deepseek"}})
	installClinePassTestState(t, cfg)
	w := httptest.NewRecorder()
	handleClinePassAccounts(w, httptest.NewRequest(http.MethodGet, "/admin/api/clinepass/accounts", nil))
	if strings.Contains(w.Body.String(), "sk-test-secret") {
		t.Fatalf("admin account API leaked full key: %s", w.Body.String())
	}
	if !strings.Contains(w.Body.String(), "sk-test****") {
		t.Fatalf("admin account API did not return a masked key: %s", w.Body.String())
	}

	w = httptest.NewRecorder()
	handleClinePassConfig(w, httptest.NewRequest(http.MethodGet, "/admin/api/clinepass/config", nil))
	if strings.Contains(w.Body.String(), "sk-test-secret") || strings.Contains(w.Body.String(), "apiKey") {
		t.Fatalf("admin config API exposed key material: %s", w.Body.String())
	}
}
