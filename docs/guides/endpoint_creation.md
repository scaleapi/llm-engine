When creating a model endpoint, you can periodically poll the model status field to
track the status of your model endpoint. In general, you'll need to wait after the 
model creation step for the model endpoint to be ready and available for use.
An example is provided below: 


```
model_name = "test_deploy"
model = Model.create(name=model_name, model="llama-2-7b", inference_frame_image_tag="0.9.4")
response = Model.get(model_name)
while response.status.name != "READY":
    print(response.status.name)
    time.sleep(60)
    response = Model.get(model_name)
```

Once the endpoint status is ready, you can use your newly created model for inference.