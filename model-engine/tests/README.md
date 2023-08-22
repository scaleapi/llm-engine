## To Run Tests:

```shell
pushd ../
PYTHONPATH=hosted_model_inference WORKSPACE=. python3 -m pytest hosted_model_inference/tests --cov=hosted_model_inference
popd
```
