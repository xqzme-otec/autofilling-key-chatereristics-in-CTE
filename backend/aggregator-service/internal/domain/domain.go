// Package domain holds the entities of the analysis dialog and the errors shared by all layers.
package domain

import "time"

// AnalysisResult is the part of ml.pipeline.Pipeline.analyze() output the aggregator uses.
// JSON tags follow the model service contract (docs/api.md, section 4); unknown fields are dropped.
type AnalysisResult struct {
	Input           *string          `json:"input"`
	Text            *string          `json:"text"`
	Corrections     []Correction     `json:"corrections"`
	Characteristics []Characteristic `json:"characteristics"`
	Candidates      []Candidate      `json:"candidates"`
	CandidatesTotal int              `json:"candidates_total"`
	RetrievalCount  *int             `json:"retrieval_count,omitempty"`
	CompatibleCount *int             `json:"compatible_count,omitempty"`
	RemainingCount  *int             `json:"remaining_count,omitempty"`
	ProductIntent   *string          `json:"product_intent,omitempty"`
	SubtypeContext  []ContextSignal  `json:"subtype_context,omitempty"`
	Errors          []AnalysisError  `json:"errors"`
	MissingRequired []string         `json:"missing_required"` // required KTRU characteristics the text lacks
	NextStep        *NextStep        `json:"next_step"`
	NextQuestion    *NextStep        `json:"next_question,omitempty"`
	Ktru            *string          `json:"ktru"`
	KtruName        *string          `json:"ktru_name"`
	Steps           []Step           `json:"steps"`
}

type Correction struct {
	From   string  `json:"from"`
	To     string  `json:"to"`
	Reason *string `json:"reason"`
}

type Characteristic struct {
	Name             string   `json:"name"`
	Value            string   `json:"value"`
	Unit             *string  `json:"unit,omitempty"`
	Source           *string  `json:"source,omitempty"`
	Origin           string   `json:"origin"`     // ktru, portal, new or answer: the source hierarchy
	Confidence       *float64 `json:"confidence"` // 0..1
	UserText         *string  `json:"user_text,omitempty"`
	UsedForNarrowing *bool    `json:"used_for_narrowing,omitempty"`
	UsedForReranking *bool    `json:"used_for_reranking,omitempty"`
}

type ContextSignal struct {
	Value            string `json:"value"`
	Source           string `json:"source"`
	UserText         string `json:"user_text"`
	UsedForNarrowing bool   `json:"used_for_narrowing"`
	UsedForReranking bool   `json:"used_for_reranking"`
}

type Candidate struct {
	Code  string   `json:"code"`
	Name  string   `json:"name"`
	Score *float64 `json:"score"`
}

type AnalysisError struct {
	Type           string  `json:"type"`
	Severity       string  `json:"severity"` // error, warning or info
	Message        string  `json:"message"`
	Characteristic *string `json:"characteristic,omitempty"`
	Suggestion     *string `json:"suggestion,omitempty"`
}

// NextStep is the clarifying question; Type is one of the Answer* constants.
type NextStep struct {
	Type           string   `json:"type"`
	Message        *string  `json:"message"`
	Characteristic *string  `json:"characteristic"`
	Options        []string `json:"options"`
}

type Step struct {
	Stage string `json:"stage"`
	Count int    `json:"count"`
}

const (
	AnswerProduct        = "product"
	AnswerCharacteristic = "characteristic"
	AnswerCode           = "code"
)

type Answer struct {
	Type           string `json:"type"`
	Value          string `json:"value"`
	Characteristic string `json:"characteristic,omitempty"`
}

type Session struct {
	ID        string         `json:"id"`
	Text      string         `json:"text"` // as typed by the user, sent to the model
	Result    AnalysisResult `json:"result"`
	Answers   []Answer       `json:"answers"`
	CreatedAt time.Time      `json:"created_at"`
	UpdatedAt time.Time      `json:"updated_at"`
}

type Status string

const (
	StatusNeedsAnswer Status = "needs_answer" // the model asks a clarifying question
	StatusDone        Status = "done"         // the final KTRU code is determined
	StatusNotFound    Status = "not_found"    // no code and nothing to ask
)

func (s *Session) Status() Status {
	switch {
	case s.Result.Ktru != nil:
		return StatusDone
	case s.Result.NextStep != nil:
		return StatusNeedsAnswer
	}
	return StatusNotFound
}
