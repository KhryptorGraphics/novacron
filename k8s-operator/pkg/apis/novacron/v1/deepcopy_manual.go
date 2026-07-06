package v1

import "k8s.io/apimachinery/pkg/runtime"

// Manual deepcopy implementations for types containing interface{} fields,
// which controller-gen cannot generate for. These types are marked
// "object:generate=false" in their declarations.

func deepCopyIfaceMap(in map[string]interface{}) map[string]interface{} {
	if in == nil {
		return nil
	}
	out := make(map[string]interface{}, len(in))
	for k, v := range in {
		out[k] = runtime.DeepCopyJSONValue(v)
	}
	return out
}

// DeepCopyInto copies the receiver into out.
func (in *AIModelConfig) DeepCopyInto(out *AIModelConfig) {
	*out = *in
	if in.TrainingConfig != nil {
		out.TrainingConfig = in.TrainingConfig.DeepCopy()
	}
	out.Parameters = deepCopyIfaceMap(in.Parameters)
}

// DeepCopy creates a new AIModelConfig.
func (in *AIModelConfig) DeepCopy() *AIModelConfig {
	if in == nil {
		return nil
	}
	out := new(AIModelConfig)
	in.DeepCopyInto(out)
	return out
}

// DeepCopyInto copies the receiver into out.
func (in *TrainingConfig) DeepCopyInto(out *TrainingConfig) {
	*out = *in
	if in.Features != nil {
		out.Features = make([]string, len(in.Features))
		copy(out.Features, in.Features)
	}
	out.Hyperparameters = deepCopyIfaceMap(in.Hyperparameters)
}

// DeepCopy creates a new TrainingConfig.
func (in *TrainingConfig) DeepCopy() *TrainingConfig {
	if in == nil {
		return nil
	}
	out := new(TrainingConfig)
	in.DeepCopyInto(out)
	return out
}

// DeepCopyInto copies the receiver into out.
func (in *Constraint) DeepCopyInto(out *Constraint) {
	*out = *in
	if in.Value != nil {
		out.Value = runtime.DeepCopyJSONValue(in.Value)
	}
}

// DeepCopy creates a new Constraint.
func (in *Constraint) DeepCopy() *Constraint {
	if in == nil {
		return nil
	}
	out := new(Constraint)
	in.DeepCopyInto(out)
	return out
}

// DeepCopyInto copies the receiver into out.
func (in *SchedulingObjective) DeepCopyInto(out *SchedulingObjective) {
	*out = *in
	if in.Target != nil {
		out.Target = runtime.DeepCopyJSONValue(in.Target)
	}
	if in.Constraints != nil {
		out.Constraints = make([]Constraint, len(in.Constraints))
		for i := range in.Constraints {
			in.Constraints[i].DeepCopyInto(&out.Constraints[i])
		}
	}
}

// DeepCopy creates a new SchedulingObjective.
func (in *SchedulingObjective) DeepCopy() *SchedulingObjective {
	if in == nil {
		return nil
	}
	out := new(SchedulingObjective)
	in.DeepCopyInto(out)
	return out
}

// DeepCopyInto copies the receiver into out.
func (in *PlacementDecision) DeepCopyInto(out *PlacementDecision) {
	*out = *in
	in.Resources.DeepCopyInto(&out.Resources)
	out.ExpectedPerformance = deepCopyIfaceMap(in.ExpectedPerformance)
}

// DeepCopy creates a new PlacementDecision.
func (in *PlacementDecision) DeepCopy() *PlacementDecision {
	if in == nil {
		return nil
	}
	out := new(PlacementDecision)
	in.DeepCopyInto(out)
	return out
}

// DeepCopyInto copies the receiver into out.
func (in *TemplateParameter) DeepCopyInto(out *TemplateParameter) {
	*out = *in
	if in.DefaultValue != nil {
		out.DefaultValue = runtime.DeepCopyJSONValue(in.DefaultValue)
	}
}

// DeepCopy creates a new TemplateParameter.
func (in *TemplateParameter) DeepCopy() *TemplateParameter {
	if in == nil {
		return nil
	}
	out := new(TemplateParameter)
	in.DeepCopyInto(out)
	return out
}
