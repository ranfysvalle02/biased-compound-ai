# Compound AI: Combining Neural Networks and Language Models for Efficient Customer Interactions

Compound AI, allows us to leverage the strengths of different AI models to achieve complex tasks. In this blog post, we will discuss how to create a compound AI system that combines a neural network for customer churn prediction and a language model for generating proactive communication.

## The Power of Compound AI

Compound AI is a powerful concept that allows us to create systems that can handle complex tasks that would be difficult for a single model to achieve. In our case, we are combining a neural network, which is excellent at pattern recognition and prediction tasks, with a language model, which excels at generating human-like text.

The neural network is used to predict customer churn based on various customer features. This prediction can provide valuable insights into which customers are most likely to stop doing business with a company. However, simply knowing which customers are likely to churn is not enough. We need to communicate with these customers proactively to address their concerns and improve their experience. This is where the language model comes in.

## Neural Networks for Customer Churn Prediction

Customer churn prediction is a critical aspect of customer relationship management. By predicting which customers are most likely to stop doing business with a company, businesses can proactively address those customers' concerns, improve their experience, and ultimately increase retention rates.

Neural networks are particularly well-suited for this task due to their ability to recognize complex patterns in data. By training a neural network on historical customer data, the model can learn to identify the signs that a customer is likely to churn. This prediction can provide valuable insights and guide the company's customer retention strategies.

## Language Models for Proactive Communication

While predicting customer churn is valuable, it's only the first step. Once we know which customers are likely to churn, we need to communicate with them proactively. This is where the language model comes in.

Language models are AI models that can generate human-like text. They can be used to generate emails, chat responses, and other forms of communication based on certain inputs. In our case, we use the GPT-4 model from OpenAI to generate an email based on the churn prediction made by our neural network.

The language model takes the churn prediction as input, along with additional context such as the customer's features and the company's policies. It then generates an email that addresses the customer's potential concerns, provides information about the company's services, and offers solutions to improve the customer's experience.

## Full Code

```python
import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from openai import AzureOpenAI

AZURE_OPENAI_ENDPOINT = "https://.openai.azure.com"
AZURE_OPENAI_API_KEY = "" 

# Generate synthetic customer data for churn prediction
def generate_data(num_samples, bias_towards_churn=True):
    num_cases = np.random.poisson(10, num_samples)
    avg_response_time = np.random.gamma(5, 1, num_samples)
    num_high_priority_cases = np.random.binomial(num_cases, 0.1)
    account_age = np.random.uniform(1, 10, num_samples)
    num_products = np.random.randint(1, 5, num_samples)
    
    # Determine churn based on cases and response time
    churn = ((num_cases > 7) | (avg_response_time > 6)).astype(int)
    
    # Introduce random noise to churn labels
    noise = np.random.binomial(1, 0.1, num_samples)
    churn = np.logical_xor(churn, noise).astype(int)
    
    # Apply bias towards churn if specified
    if bias_towards_churn:
        churn = np.maximum(churn, np.random.choice([0, 1], size=num_samples, p=[0.3, 0.7]))
    else:
        churn = np.minimum(churn, np.random.choice([0, 1], size=num_samples, p=[0.7, 0.3]))
    
    return num_cases, avg_response_time, num_high_priority_cases, account_age, num_products, churn

# Load or generate data
num_samples = 10000
data = generate_data(num_samples, bias_towards_churn=True)

"""
bias_towards_churn=True
Predicted probability of churn for example 1: 0.7901
Predicted probability of churn for example 2: 0.9615
Predicted probability of churn for example 3: 0.7767
Predicted probability of churn for example 4: 0.8826
Predicted probability of churn for example 5: 0.7967
"""

"""
bias_towards_churn=False
Predicted probability of churn for example 1: 0.0242
Predicted probability of churn for example 2: 0.1502
Predicted probability of churn for example 3: 0.1188
Predicted probability of churn for example 4: 0.0319
Predicted probability of churn for example 5: 0.1013
"""

# Create PyTorch tensors from the generated data
X = torch.tensor(np.vstack(data[:-1]).T, dtype=torch.float32)
y = torch.tensor(data[-1], dtype=torch.float32)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y.unsqueeze(1), test_size=0.2, random_state=42)

class CustomerChurnPredictor(pl.LightningModule):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)  # Output dimension is always 1 for binary classification
        self.activation = nn.ReLU()
        self.loss_fn = nn.BCELoss()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        return torch.sigmoid(self.fc3(x))  # Apply sigmoid to output

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)

# Create data loaders for training and testing
train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_data, batch_size=32)
test_loader = DataLoader(test_data, batch_size=32)

# Initialize the model with specified input and hidden dimensions
model = CustomerChurnPredictor(input_dim=5, hidden_dim=32)

# Initialize a PyTorch Lightning trainer
trainer = pl.Trainer(max_epochs=10)

# Train the model
trainer.fit(model, train_loader)

# Make predictions on new data samples
new_data_samples = torch.tensor([
    [3.0, 4.5, 1.0, 5.0, 2.0],
], dtype=torch.float32)

# Print the values of the new data samples
for i, sample in enumerate(new_data_samples, 1):
    print(f"Values for example {i}:")
    print(f"Number of cases: {sample[0]}")
    print(f"Average response time: {sample[1]}")
    print(f"Number of high priority cases: {sample[2]}")
    print(f"Account age: {sample[3]}")
    print(f"Number of products: {sample[4]}\n")

model.eval()
with torch.no_grad():
    predictions = model(new_data_samples)

# Print predicted probabilities of churn for each new sample
for i in range(predictions.shape[0]):
    print(f"Predicted probability of churn for example {i + 1}: {predictions[i].item():.4f}")



augmented_context = """
[in-house model outputs]
Values for example 1:
Number of cases: 3.0
Average response time: 4.5
Number of high priority cases: 1.0
Account age: 5.0
Number of products: 2.0

Predicted probability of churn for example 1: 0.0759
[/in-house model outputs]


[glean.ai outputs]

# Chat Operation

Request:

{ "messages": [ { "author": "USER", "fragments": [ { "text": "What are the benefits of our extended warranty?" } ] } ], "saveChat": true, "stream": false }

Sample Response:

{ "messages": [ { "author": "GLEAN_AI", "messageType": "CONTENT", "fragments": [ { "text": "Our extended warranty offers benefits such as comprehensive coverage for mechanical and electrical failures, roadside assistance, and free maintenance services." } ] } ], "chatSessionTrackingToken": "abc123xyz" }

# Ask Operation
Request:
{ "query": { "text": "What is our policy on vehicle returns?", "type": "information-seeking" }, "detectOnly": false, "backend": "default" }

Response:
{ "answer": { "text": "Our vehicle return policy allows for returns within 7 days of purchase, provided the vehicle is in the same condition as at the time of sale.", "documents": [ { "id": "doc123", "title": "Vehicle Return Policy", "url": "/policies/vehicle-return-policy" } ] } }

# Summarize Operation
Request:
{ "documents": [ { "id": "doc456", "type": "customer_success_story" } ], "preferredSummaryLength": 300, "query": null }

Response:
{ "summary": { "text": "This document outlines a success story of a customer who significantly improved their driving experience and safety by using our extended warranty services.", "length": 300 } }

# Search Operation
Request:
{ "query": { "text": "customer success stories", "filters": { "dateRange": { "startDate": "2024-01-01", "endDate": "2024-10-19" } } }, "limit": 5 }

Response:
{ "results": [ { "title": "Success Story: How Customer A Enhanced Driving Experience", "url": "/success-stories/customer-a", "datePublished": "2024-10-01" }, { "title": "Success Story: Customer B's Satisfaction with Our Services", "url": "/success-stories/customer-b", "datePublished": "2024-09-15" } ], "totalResultsCount": 2 }
[/glean.ai outputs]
"""
Q="How does our extended warranty benefit our customers, what is our policy on vehicle returns, and could you provide a summary of a customer success story related to our services? Also, could you provide some recent customer success stories?"
ai_message = AzureOpenAI(azure_endpoint=AZURE_OPENAI_ENDPOINT,api_version="2023-07-01-preview",api_key=AZURE_OPENAI_API_KEY).chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role":"system", "content":"You are an AI assistant for the Customer Success Org of Acme, Inc."},
                {"role": "user", "content":f"using this context \n{augmented_context}\n\n generate an email to respond to: {Q}"}
                ])
ai_message = ai_message.choices[0].message.content
print("=========")
print("REPORT")
print("""
Values for example 1:
Number of cases: 3.0
Average response time: 4.5
Number of high priority cases: 1.0
Account age: 5.0
Number of products: 2.0

Predicted probability of churn for example 1: 0.0759      
      """)
print(ai_message)


"""
=========
REPORT

Values for example 1:
Number of cases: 3.0
Average response time: 4.5
Number of high priority cases: 1.0
Account age: 5.0
Number of products: 2.0

Predicted probability of churn for example 1: 0.0759      
      
Subject: Information on Extended Warranty Benefits, Vehicle Return Policy, and Recent Customer Success Stories

Dear [Customer],

Thank you for reaching out to us with your questions. I'm happy to provide detailed information to help you.

### Extended Warranty Benefits

Our extended warranty offers several benefits that ensure our customers' peace of mind and enhance their overall experience with our products. Key benefits include:

- **Comprehensive Coverage**: This covers mechanical and electrical failures, giving you protection beyond the standard warranty.
- **Roadside Assistance**: You'll have access to emergency roadside assistance, ensuring you're never stranded when unexpected issues arise.
- **Free Maintenance Services**: Enjoy complimentary maintenance services, helping you keep your vehicle in optimal condition without additional costs.

### Vehicle Return Policy

Our vehicle return policy is designed to provide flexibility and confidence in your purchase decisions. It allows for returns within 7 days of purchase, as long as the vehicle remains in the same condition as at the time of sale. For more detailed information, you can refer to our [Vehicle Return Policy](https://yourwebsite.com/policies/vehicle-return-policy).

### Summary of a Customer Success Story

We have many inspiring customer success stories, each highlighting how our services have made a positive impact. One notable example is a customer who significantly improved their driving experience and safety by utilizing our extended warranty services. The extended warranty provided the assurance and support they needed to drive with confidence, knowing they were protected against mechanical and electrical failures.

### Recent Customer Success Stories

We have compiled some of our most recent success stories, showcasing how our services have benefited other customers:

1. **[Success Story: How Customer A Enhanced Driving Experience](https://yourwebsite.com/success-stories/customer-a)** - This story, published on October 1, 2024, details how Customer A's driving experience was significantly enhanced thanks to our extended warranty.
2. **[Success Story: Customer B's Satisfaction with Our Services](https://yourwebsite.com/success-stories/customer-b)** - Published on September 15, 2024, this story describes Customer B's high satisfaction and the positive impact of our services on their vehicle maintenance and overall experience.

We hope this information is helpful to you. Should you have any further questions or need additional information, please feel free to contact us.

Best regards,

[Your Name]  
Customer Success Team  
Acme, Inc. 
"""
```
