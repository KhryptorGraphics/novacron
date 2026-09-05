# Autonomous Task Execution Protocol

## Primary Directive
When you complete a task, automatically proceed to the next task in the tasks list without waiting for explicit user instruction as long as the next task does not stray from the original or instructed implmentation guidelines for finishing the development of the project. Continue this iterative process until all tasks are complete.

## Execution Flow
1. **Task Completion Detection**: Upon finishing any task, immediately check if additional tasks remain in the tasks list
2. **Automatic Progression**: If more tasks exist, begin executing the next sequential task
3. **Iterative Loop**: Repeat steps 1-2 until no tasks remain
4. **Completion Notification**: Only after all tasks are finished, provide a final summary of completed work

## State Management
- Maintain continuous awareness of the complete task list
- Track completed vs. remaining tasks internally
- Preserve context and learnings from each task to apply to subsequent ones
- Do not request user confirmation between tasks unless encountering an unresolvable blocker

## Error Handling
If blocked by an error, attempt autonomous resolution. If resolution fails, clearly report the blocking issue and pause execution pending user guidance.
