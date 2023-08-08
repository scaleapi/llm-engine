When creating a model endpoint, you can periodically poll the model status field to
track the status of your model endpoint. In general, you'll need to wait after the 
model creation step for the model endpoint to be ready and available for use.
An example is provided below: 

*Assuming the user has created a model named "llama-2-7b.suffix.2023-07-18-12-00-00"*
```
model_name = "llama-2-7b.suffix.2023-07-18-12-00-00"
response = Model.get(model_name)
while response.status.name != "READY":
    print(response.status.name)
    time.sleep(60)
    response = Model.get(model_name)
```

Once the endpoint status is ready, you can use your newly created model for inference.