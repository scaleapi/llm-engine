# Fine-tuning

Learn how to customize your models on your data with fine-tuning. Or get started right away with our [fine-tuning cookbook](https://github.com/scaleapi/llm-engine/blob/main/docs/examples/finetuning.ipynb).

## Introduction

Fine-tuning helps improve model performance by training on specific examples of prompts and desired responses. LLMs are initially trained on data collected from the entire internet. With fine-tuning, LLMs can be optimized to perform better in a specific domain by learning from examples for that domain. Smaller LLMs that have been fine-tuned on a specific use case [often outperform](https://arxiv.org/abs/2305.15334) larger ones that were trained more generally.

Fine-tuning allows for:

1. Higher quality results than prompt engineering alone
2. Cost savings through shorter prompts
3. The ability to reach equivalent accuracy with a smaller model
4. Lower latency at inference time
5. The chance to show an LLM more examples than can fit in a single context window

LLM Engine's fine-tuning API lets you fine-tune various open source LLMs on your own data and then make inference calls to the resulting LLM. For more specific details, see the [fine-tuning API reference](../../api/python_client/#llmengine.FineTune).

## Producing high quality data for fine-tuning

The training data for fine-tuning should consist of prompt and response pairs.

As a rule of thumb, you should expect to see linear improvements in your fine-tuned model's quality with each doubling of the dataset size. Having high-quality data is also essential to improving performance. For every linear increase in the error rate in your training data, you may encounter a roughly quadratic increase in your fine-tuned model's error rate.

High quality data is critical to achieve improved model performance, and in several cases will require _experts_ to
generate and prepare data - the breadth and diversity of the data is highly critical. Scale's Data Engine can help
prepare such high quality, diverse data sets - more information [here](https://scale.com/rlhf).

## Preparing data

Your data must be formatted as a CSV file that includes two columns: `prompt` and `response`. A maximum of 100,000 rows of data is currently supported. At least 200 rows of data is recommended to start to see benefits from fine-tuning.

For better model performance, we recommend adding the same separator tokens to the end of the prompts. For example `Answer:\n`.

Here is an example script to create a 50-row CSV of properly formatted data for fine-tuning an airline question answering bot

<details>
<summary>Creating a sample dataset</summary>

```python
import csv

# Define data
data = [
    ("What is your policy on carry-on luggage?Answer:\n", "Our policy allows each passenger to bring one piece of carry-on luggage and one personal item such as a purse or briefcase. The maximum size for carry-on luggage is 22 x 14 x 9 inches."),
    ("How can I change my flight?Answer:\n", "You can change your flight through our website or mobile app. Go to 'Manage my booking' section, enter your booking reference and last name, then follow the prompts to change your flight."),
    ("What meals are available on my flight?Answer:\n", "We offer a variety of meals depending on the flight's duration and route. These can range from snacks and light refreshments to full-course meals on long-haul flights. Specific meal options can be viewed during the booking process."),
    ("How early should I arrive at the airport before my flight?Answer:\n", "We recommend arriving at least two hours before domestic flights and three hours before international flights."),
    ("Can I select my seat in advance?Answer:\n", "Yes, you can select your seat during the booking process or afterwards via the 'Manage my booking' section on our website or mobile app."),
    ("What should I do if my luggage is lost?Answer:\n", "If your luggage is lost, please report this immediately at our 'Lost and Found' counter at the airport. We will assist you in tracking your luggage."),
    ("Do you offer special assistance for passengers with disabilities?Answer:\n", "Yes, we offer special assistance for passengers with disabilities. Please notify us of your needs at least 48 hours prior to your flight."),
    ("Can I bring my pet on the flight?Answer:\n", "Yes, we allow small pets in the cabin, and larger pets in the cargo hold. Please check our pet policy for more details."),
    ("What is your policy on flight cancellations?Answer:\n", "In case of flight cancellations, we aim to notify passengers as early as possible and offer either a refund or a rebooking on the next available flight."),
    ("Can I get a refund if I cancel my flight?Answer:\n", "Refunds depend on the type of ticket purchased. Please check our cancellation policy for details. Non-refundable tickets, however, are typically not eligible for refunds unless due to extraordinary circumstances."),
    ("How can I check-in for my flight?Answer:\n", "You can check-in for your flight either online, through our mobile app, or at the airport. Online and mobile app check-in opens 24 hours before departure and closes 90 minutes before."),
    ("Do you offer free meals on your flights?Answer:\n", "Yes, we serve free meals on all long-haul flights. For short-haul flights, we offer a complimentary drink and snack. Special meal requests should be made at least 48 hours before departure."),
    ("Can I use my electronic devices during the flight?Answer:\n", "Small electronic devices can be used throughout the flight in flight mode. Larger devices like laptops may be used above 10,000 feet."),
    ("How much baggage can I check-in?Answer:\n", "The checked baggage allowance depends on the class of travel and route. The details would be mentioned on your ticket, or you can check on our website."),
    ("How can I request for a wheelchair?Answer:\n", "To request a wheelchair or any other special assistance, please call our customer service at least 48 hours before your flight."),
    ("Do I get a discount for group bookings?Answer:\n", "Yes, we offer discounts on group bookings of 10 or more passengers. Please contact our group bookings team for more information."),
    ("Do you offer Wi-fi on your flights?Answer:\n", "Yes, we offer complimentary Wi-fi on select flights. You can check the availability during the booking process."),
    ("What is the minimum connecting time between flights?Answer:\n", "The minimum connecting time varies depending on the airport and whether your flight is international or domestic. Generally, it's recommended to allow at least 45-60 minutes for domestic connections and 60-120 minutes for international."),
    ("Do you offer duty-free shopping on international flights?Answer:\n", "Yes, we have a selection of duty-free items that you can pre-order on our website or purchase onboard on international flights."),
    ("Can I upgrade my ticket to business class?Answer:\n", "Yes, you can upgrade your ticket through the 'Manage my booking' section on our website or by contacting our customer service. The availability and costs depend on the specific flight."),
    ("Can unaccompanied minors travel on your flights?Answer:\n", "Yes, we do accommodate unaccompanied minors on our flights, with special services to ensure their safety and comfort. Please contact our customer service for more details."),
    ("What amenities do you provide in business class?Answer:\n", "In business class, you will enjoy additional legroom, reclining seats, premium meals, priority boarding and disembarkation, access to our business lounge, extra baggage allowance, and personalized service."),
    ("How much does extra baggage cost?Answer:\n", "Extra baggage costs vary based on flight route and the weight of the baggage. Please refer to our 'Extra Baggage' section on the website for specific rates."),
    ("Are there any specific rules for carrying liquids in carry-on?Answer:\n", "Yes, liquids carried in your hand luggage must be in containers of 100 ml or less and they should all fit into a single, transparent, resealable plastic bag of 20 cm x 20 cm."),
    ("What if I have a medical condition that requires special assistance during the flight?Answer:\n", "We aim to make the flight comfortable for all passengers. If you have a medical condition that may require special assistance, please contact our ‘special services’ team 48 hours before your flight."),
    ("What in-flight entertainment options are available?Answer:\n", "We offer a range of in-flight entertainment options including a selection of movies, TV shows, music, and games, available on your personal seat-back screen."),
    ("What types of payment methods do you accept?Answer:\n", "We accept credit/debit cards, PayPal, bank transfers, and various other forms of payment. The available options may vary depending on the country of departure."),
    ("How can I earn and redeem frequent flyer miles?Answer:\n", "You can earn miles for every journey you take with us or our partner airlines. These miles can be redeemed for flight tickets, upgrades, or various other benefits. To earn and redeem miles, you need to join our frequent flyer program."),
    ("Can I bring a stroller for my baby?Answer:\n", "Yes, you can bring a stroller for your baby. It can be checked in for free and will normally be given back to you at the aircraft door upon arrival."),
    ("What age does my child have to be to qualify as an unaccompanied minor?Answer:\n", "Children aged between 5 and 12 years who are traveling alone are considered unaccompanied minors. Our team provides special care for these children from departure to arrival."),
    ("What documents do I need to travel internationally?Answer:\n", "For international travel, you need a valid passport and may also require visas, depending on your destination and your country of residence. It's important to check the specific requirements before you travel."),
    ("What happens if I miss my flight?Answer:\n", "If you miss your flight, please contact our customer service immediately. Depending on the circumstances, you may be able to rebook on a later flight, but additional fees may apply."),
    ("Can I travel with my musical instrument?Answer:\n", "Yes, small musical instruments can be brought on board as your one carry-on item. Larger instruments must be transported in the cargo, or if small enough, a seat may be purchased for them."),
    ("Do you offer discounts for children or infants?Answer:\n", "Yes, children aged 2-11 traveling with an adult usually receive a discount on the fare. Infants under the age of 2 who do not occupy a seat can travel for a reduced fare or sometimes for free."),
    ("Is smoking allowed on your flights?Answer:\n", "No, all our flights are non-smoking for the comfort and safety of all passengers."),
    ("Do you have family seating?Answer:\n", "Yes, we offer the option to seat families together. You can select seats during booking or afterwards through the 'Manage my booking' section on the website."),
    ("Is there any discount for senior citizens?Answer:\n", "Some flights may offer a discount for senior citizens. Please check our website or contact customer service for accurate information."),
    ("What items are prohibited on your flights?Answer:\n", "Prohibited items include, but are not limited to, sharp objects, firearms, explosive materials, and certain chemicals. You can find a comprehensive list on our website under the 'Security Regulations' section."),
    ("Can I purchase a ticket for someone else?Answer:\n", "Yes, you can purchase a ticket for someone else. You'll need their correct name as it appears on their government-issued ID, and their correct travel dates."),
    ("What is the process for lost and found items on the plane?Answer:\n", "If you realize you forgot an item on the plane, report it as soon as possible to our lost and found counter. We will make every effort to locate and return your item."),
    ("Can I request a special meal?Answer:\n", "Yes, we offer a variety of special meals to accommodate dietary restrictions. Please request your preferred meal at least 48 hours prior to your flight."),
    ("Is there a weight limit for checked baggage?Answer:\n", "Yes, luggage weight limits depend on your ticket class and route. You can find the details on your ticket or by visiting our website."),
    ("Can I bring my sports equipment?Answer:\n", "Yes, certain types of sports equipment can be carried either as or in addition to your permitted baggage. Some equipment may require additional fees. It's best to check our policy on our website or contact us directly."),
    ("Do I need a visa to travel to certain countries?Answer:\n", "Yes, visa requirements depend on the country you are visiting and your nationality. We advise checking with the relevant embassy or consulate prior to travel."),
    ("How can I add extra baggage to my booking?Answer:\n", "You can add extra baggage to your booking through the 'Manage my booking' section on our website or by contacting our customer services."),
    ("Can I check-in at the airport?Answer:\n", "Yes, you can choose to check-in at the airport. However, we also offer online and mobile check-in, which may save you time."),
    ("How do I know if my flight is delayed or cancelled?Answer:\n", "In case of any changes to your flight, we will attempt to notify all passengers using the contact information given at the time of booking. You can also check your flight status on our website."),
    ("What is your policy on pregnant passengers?Answer:\n", "Pregnant passengers can travel up to the end of the 36th week for single pregnancies, and the end of the 32nd week for multiple pregnancies. We recommend consulting your doctor before any air travel."),
    ("Can children travel alone?Answer:\n", "Yes, children age 5 to 12 can travel alone as unaccompanied minors. We provide special care for these seats. Please contact our customer service for more information."),
    ("How can I pay for my booking?Answer:\n", "You can pay for your booking using a variety of methods including credit and debit cards, PayPal, or bank transfers. The options may vary depending on the country of departure."),
]

# Write data to a CSV file
with open('customer_service_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["prompt", "response"])
    writer.writerows(data)
```

</details>

## Making your data accessible to LLM Engine

Currently, data needs to be uploaded to a publicly accessible web URL so that it can be read
for fine-tuning. Publicly accessible HTTP and HTTPS URLs are currently supported.
Support for privately sharing data with the LLM Engine API is coming shortly. For quick
iteration, you can look into tools like Pastebin or GitHub Gists to quickly host your CSV
files in a public manner. An example Github Gist can be found
[here](https://gist.github.com/yunfeng-scale/275241bfe782887eabced8147bf30315). To use the gist,
you can use the URL given when you click the “Raw” button
([URL](https://gist.githubusercontent.com/yunfeng-scale/275241bfe782887eabced8147bf30315/raw/7da46e2f0e45748555a7a8b811c6c8ef13b96947/llm_sample_data.csv)).

## Launching the fine-tune

Once you have uploaded your data, you can use the LLM Engine's [FineTune.Create](../../api/python_client/#llmengine.fine_tuning.FineTune.create) API to launch a fine-tune. You will need to specify which base model to fine-tune, the locations of the training file and optional validation data file, an optional set of hyperparameters to customize the fine-tuning behavior, and an optional suffix to append to the name of the fine-tune. For sequences longer than the native
`max_seq_length` of the model, the sequences will be truncated.

If you specify a suffix, the fine-tune will be named `model.suffix.<timestamp>`. If you do not,
the fine-tune will be named `model.<timestamp>`. The timestamp will be the time the fine-tune was
launched.

<details>
<summary>Hyper-parameters for fine-tune</summary>

- `lr`: Peak learning rate used during fine-tuning. It decays with a cosine schedule afterward. (Default: 2e-3)
- `warmup_ratio`: Ratio of training steps used for learning rate warmup. (Default: 0.03)
- `epochs`: Number of fine-tuning epochs. This should be less than 20. (Default: 5)
- `weight_decay`: Regularization penalty applied to learned weights. (Default: 0.001)
</details>

=== "Create a fine-tune in python"

```python
from llmengine import FineTune

response = FineTune.create(
    model="llama-2-7b",
    training_file="s3://my-bucket/path/to/training-file.csv",
)

print(response.json())
```

See the [Model Zoo](../../model_zoo) to see which models have fine-tuning support.

Once the fine-tune is launched, you can also [get the status of your fine-tune](../../api/python_client/#llmengine.fine_tuning.FineTune.get). You can also [list events that your fine-tune produces](../../api/python_client/#llmengine.fine_tuning.FineTune.get_events).

## Making inference calls to your fine-tune

Once your fine-tune is finished, you will be able to start making inference requests to the
model. You can use the `fine_tuned_model` returned from your
[FineTune.get](../../api/python_client/#llmengine.fine_tuning.FineTune.get)
API call to reference your fine-tuned model in the Completions API. Alternatively, you can list
available LLMs with `Model.list` in order to find the name of your fine-tuned model. See the
[Completion API](../../api/python_client/#llmengine.Completion) for more details. You can then
use that name to direct your completion requests. You must wait until your fine-tune is complete
before you can plug it into the Completions API. You can check the status of your fine-tune with
[FineTune.get](../../api/python_client/#llmengine.fine_tuning.FineTune.get). If separator tokens
were used during fine-tune process, you should include them at inference time as well.

=== "Inference with a fine-tuned model in python"

```python
from llmengine import Completion

response = Completion.create(
    model="llama-2-7b.airlines.2023-07-17-08-30-45",
    prompt="Do you offer in-flight Wi-fi?Answer:\n",
    max_new_tokens=100,
    temperature=0.2,
)
print(response.json())
```
