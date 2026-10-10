// Package service implements the analysis dialog: validate and preprocess the text -> ask the
// model -> keep the session for follow-up answers.
//
// The model service is stateless: every call receives the full text and all answers given so far.
// The session keeps them so the frontend only sends the next answer.
package service

import (
	"context"
	"crypto/rand"
	"errors"
	"log/slog"
	"slices"
	"time"
	"unicode/utf8"

	"github.com/xqzme-otec/autofilling-key-chatereristics-in-CTE/backend/internal/domain"
	"github.com/xqzme-otec/autofilling-key-chatereristics-in-CTE/backend/internal/repository"
)

type Model interface {
	Analyze(ctx context.Context, text string, answers []domain.Answer) (*domain.AnalysisResult, error)
}

type Analysis struct {
	sessions      repository.SessionRepository
	model         Model
	maxTextLength int
}

func NewAnalysis(sessions repository.SessionRepository, model Model, maxTextLength int) *Analysis {
	return &Analysis{sessions: sessions, model: model, maxTextLength: maxTextLength}
}

// Start creates a session from a free-form product description.
func (s *Analysis) Start(ctx context.Context, rawText string) (*domain.Session, error) {
	if utf8.RuneCountInString(rawText) > s.maxTextLength {
		return nil, domain.Errorf(domain.CodeInvalidText, "Text is longer than %d characters", s.maxTextLength)
	}
	text := preprocess(rawText)
	if text == "" {
		return nil, domain.Errorf(domain.CodeInvalidText, "Text must not be empty")
	}

	result, err := s.model.Analyze(ctx, text, []domain.Answer{})
	if err != nil {
		return nil, err
	}
	now := time.Now().UTC()
	session := &domain.Session{ID: rand.Text(), RawText: rawText, Text: text, Result: *result,
		Answers: []domain.Answer{}, CreatedAt: now, UpdatedAt: now}
	if err := s.sessions.Save(ctx, session); err != nil {
		return nil, err
	}
	slog.Info("session started", "id", session.ID, "status", session.Status(), "candidates", result.CandidatesTotal)
	return session, nil
}

// Answer answers the pending question of the session with one of its options.
func (s *Analysis) Answer(ctx context.Context, id, value string) (*domain.Session, error) {
	session, err := s.sessions.Get(ctx, id)
	if errors.Is(err, repository.ErrSessionNotFound) {
		return nil, domain.Errorf(domain.CodeSessionNotFound, "Session %s not found", id)
	}
	if err != nil {
		return nil, err
	}
	answer, err := toAnswer(session.Result.NextStep, value)
	if err != nil {
		return nil, err
	}

	answers := append(slices.Clone(session.Answers), answer)
	result, err := s.model.Analyze(ctx, session.Text, answers)
	if err != nil {
		return nil, err
	}
	session.Answers, session.Result, session.UpdatedAt = answers, *result, time.Now().UTC()
	if err := s.sessions.Save(ctx, session); err != nil {
		return nil, err
	}
	slog.Info("session answered", "id", session.ID, "type", answer.Type, "status", session.Status(),
		"candidates", result.CandidatesTotal)
	return session, nil
}

// toAnswer: the frontend sends only the chosen option; the question type comes from the last next_step.
func toAnswer(q *domain.NextStep, value string) (domain.Answer, error) {
	if q == nil {
		return domain.Answer{}, domain.Errorf(domain.CodeNoPendingQuestion, "There is no pending question in this session")
	}
	if !slices.Contains(q.Options, value) {
		return domain.Answer{}, domain.Errorf(domain.CodeInvalidAnswer, "Value '%s' is not one of the offered options", value)
	}
	a := domain.Answer{Type: q.Type, Value: value}
	if q.Type == domain.AnswerCharacteristic && q.Characteristic != nil {
		a.Characteristic = *q.Characteristic
	}
	return a, nil
}
