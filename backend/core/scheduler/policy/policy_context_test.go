package policy

import (
	"testing"
)

func TestEvaluationContext(t *testing.T) {
	// Create a new evaluation context
	ctx := NewEvaluationContext()

	// Test variable operations
	ctx.SetVariable("test", "value")
	value, exists := ctx.GetVariable("test")
	if !exists {
		t.Errorf("Variable 'test' not found")
	}
	if value != "value" {
		t.Errorf("Expected 'value', got %v", value)
	}

	// Test scoring operations
	ctx.AddScore(10.0, "test score")
	score := ctx.GetScore()
	if score != 10.0 {
		t.Errorf("Expected score 10.0, got %f", score)
	}

	// Test filtering operations
	ctx.SetFiltered(true, "test filter")
	filtered, reason := ctx.IsFiltered()
	if !filtered {
		t.Errorf("Expected filtered=true, got false")
	}
	if reason != "test filter" {
		t.Errorf("Expected reason 'test filter', got '%s'", reason)
	}

	// Test initialization
	vm := map[string]interface{}{"id": "vm-001"}
	source := map[string]interface{}{"id": "source-node"}
	candidate := map[string]interface{}{"id": "candidate-node"}

	ctx.InitializeForVM(vm, source, candidate)
	if ctx.VM["id"] != "vm-001" {
		t.Errorf("VM initialization failed")
	}
	if ctx.SourceNode["id"] != "source-node" {
		t.Errorf("SourceNode initialization failed")
	}
	if ctx.CandidateNode["id"] != "candidate-node" {
		t.Errorf("CandidateNode initialization failed")
	}
}

func TestPolicyEvaluationContext(t *testing.T) {
	// Create a new policy evaluation context
	vm := map[string]interface{}{"id": "vm-001"}
	source := map[string]interface{}{"id": "source-node"}
	candidate := map[string]interface{}{"id": "candidate-node"}

	ctx := NewPolicyEvaluationContext(vm, source, candidate)

	// Test parameter operations
	ctx.SetParameter("param1", 123)
	value, exists := ctx.GetParameter("param1")
	if !exists {
		t.Errorf("Parameter 'param1' not found")
	}
	if value != 123 {
		t.Errorf("Expected 123, got %v", value)
	}

	// Test that parameter is also set as a variable with param. prefix
	varValue, exists := ctx.GetVariable("param.param1")
	if !exists {
		t.Errorf("Variable 'param.param1' not found")
	}
	if varValue != 123 {
		t.Errorf("Expected 123, got %v", varValue)
	}

	// Test attribute operations
	ctx.SetAttribute("attr1", "attr-value")
	attrValue, exists := ctx.GetAttribute("attr1")
	if !exists {
		t.Errorf("Attribute 'attr1' not found")
	}
	if attrValue != "attr-value" {
		t.Errorf("Expected 'attr-value', got %v", attrValue)
	}

	// Test that base context methods work
	ctx.SetVariable("test", "value")
	value, exists = ctx.GetVariable("test")
	if !exists {
		t.Errorf("Variable 'test' not found")
	}
	if value != "value" {
		t.Errorf("Expected 'value', got %v", value)
	}
}
