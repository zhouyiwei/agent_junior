# Solutions Folder

## Architecture overview
```
                +-----------+                  
                | __start__ |                  
                +-----------+                  
                      *                        
                      *                        
                      *                        
             +-----------------+               
             | classify_intent |               
             +-----------------+               
               **            **                
             **                **              
           **                    **            
+---------------+           +---------------+  
| score_latency |           | score_mission |  
+---------------+           +---------------+  
               **            **                
                 **        **                  
                   **    **                    
              +---------------+                
              | make_decision |                
              +---------------+                
                      *                        
                      *                        
                      *                        
                 +---------+                   
                 | __end__ |                   
                 +---------+                  
```
## Prompt design decisions for each step
Please find in the customer_router.py file.

## Key results (quality, latency, cost vs. baselines)
Please find in the evaluation.ipynb file.

## Analysis of routing overhead
Please find in the evaluation.ipynb file.

## Know limitations
* Prompts can be further optimised by including representative examples and alternative wording, based on the 
performance on the sampled dataset.
* Some logic in `make_decsion` can be explained better through functions, because they are deterministic 
and do not need any reasoning.
* Confidence score in `classify_intent` is not considered for later stages.
* In `make_decision`, different LLM's cost and performance can be considered to make a more informed decision.
* When calculating cost, routing cost is not included.
* Unit tests and integration tests.
* RateLimit and NotEnoughCredit errors from OpenRouter, and the output validation errors during batch evaluation make the evaluation results unreliable.
* Concurrency issues.
* No learning from historical experiences.


__________

This folder is where candidates should implement their routing solution.

## Structure

```
solutions/
├── README.md           # This file
├── custom_router.py    # Your main router implementation (REQUIRED)
└── ...                 # Any additional files you need
```

## Requirements

1. **Main Implementation**: Your final router must be in `custom_router.py`
2. **Class Name**: The main router class must be named `CustomRouter`
3. **Inheritance**: Must extend `BaseRouter` from `src.router`

## Adding Additional Files

You may add any additional files as needed:

- **Helper modules**: `classifier.py`, `features.py`, `utils.py`, etc.
- **Data files**: Training data, lookup tables, configuration files
- **Tests**: Unit tests for your implementation
- **Documentation**: Notes, diagrams, or analysis

## Usage

Your router will be imported and benchmarked like this:

```python
from solutions.custom_router import CustomRouter

router = CustomRouter()
results = await benchmark_router(router)
```

## Getting Started

1. Open `custom_router.py`
2. Implement the `route()` method
3. Run `uv run python smoke_test.py` to verify setup
4. Run `uv run python eval.py` to benchmark your router
