// Package model is the client of the model service (HTTP wrapper around ml.pipeline.Pipeline.analyze).
package model

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"slices"
	"time"

	"github.com/xqzme-otec/autofilling-key-chatereristics-in-CTE/backend/internal/domain"
)

const analyzePath = "/analyze"

type Client struct {
	http    *http.Client
	baseURL string
}

func NewClient(baseURL string, timeout time.Duration) *Client {
	return &Client{http: &http.Client{Timeout: timeout}, baseURL: baseURL}
}

func (c *Client) Analyze(ctx context.Context, text string, answers []domain.Answer) (*domain.AnalysisResult, error) {
	payload, err := json.Marshal(map[string]any{"raw": text, "answers": answers})
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+analyzePath, bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.http.Do(req)
	if err != nil {
		return nil, transportError(err)
	}
	defer resp.Body.Close()
	// Reading the body to the end lets the keep-alive connection be reused; the client
	// timeout also covers this read.
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, transportError(err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, domain.Errorf(domain.CodeModelBadResponse, "Model service returned HTTP %d", resp.StatusCode)
	}
	var result domain.AnalysisResult
	if err := json.Unmarshal(body, &result); err != nil {
		return nil, domain.Errorf(domain.CodeModelBadResponse, "Model service returned an unexpected response: %v", err)
	}
	if err := validate(&result); err != nil {
		return nil, domain.Errorf(domain.CodeModelBadResponse, "Model service returned an unexpected response: %v", err)
	}
	return &result, nil
}

func transportError(err error) *domain.Error {
	var ne net.Error
	if errors.As(err, &ne) && ne.Timeout() {
		return domain.Errorf(domain.CodeModelTimeout, "Model service did not respond in time")
	}
	return domain.Errorf(domain.CodeModelUnavailable, "Model service is unavailable: %v", err)
}

// validate checks the fields the aggregator relies on (docs/api.md, section 4).
func validate(r *domain.AnalysisResult) error {
	if r.Input == nil || r.Text == nil {
		return errors.New("missing input or text")
	}
	if q := r.NextStep; q != nil {
		types := []string{domain.AnswerProduct, domain.AnswerCharacteristic, domain.AnswerCode}
		if !slices.Contains(types, q.Type) || q.Message == nil || q.Options == nil {
			return errors.New("next_step requires type, message and options")
		}
	}
	return nil
}
