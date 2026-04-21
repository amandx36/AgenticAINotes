# 📊 STRUCTURED OUTPUT & PARSERS - Complete Guide

## Table of Contents
1. [Why Structured Output](#why-structured-output)
2. [Output Parser Types](#output-parser-types)
3. [Pydantic Schema Design](#pydantic-schema-design)
4. [Implementation Patterns](#implementation-patterns)
5. [Error Recovery](#error-recovery)
6. [Real-World Examples](#real-world-examples)
7. [Interview Questions](#interview-questions)

---




## Why Structured Output

### The Problem: LLMs Return Unstructured Text

```python
llm = ChatOpenAI()

response = llm.invoke("Review this product: Great phone, fast processor, battery life is good")
# Output: "Great phone, fast processor, battery life is good"
# Type: str (just a string!)

# Now what?
# - Can't extract sentiment programmatically
# - Can't use parts separately
# - Can't validate format
# - Can't pass to other systems expect structured data
```

### The Solution: Parse Into Structured Data

```python
from pydantic import BaseModel

class Review(BaseModel):
    sentiment: str  # "positive", "negative"
    rating: int     # 1-5
    pros: list[str]
    cons: list[str]

# Now:
review = parse_llm_response("Great phone, fast processor, long battery...")
# Result: Review(sentiment="positive", rating=4, pros=["fast"], cons=[])

# Can use structured data:
if review.sentiment == "positive":
    show_testimonial(review)
```

### Use Cases for Structured Output

| Use Case | Benefit |
|----------|---------|
| **Data Extraction** | Invoice → JSON with line items, total, tax |
| **API Building** | Chatbot → Structured API requests |
| **Validation** | Ensure output meets constraints |
| **Analytics** | Parse and aggregate responses |
| **Agent Tools** | Agents expect typed inputs |
| **Database Storage** | Direct insert into typed columns |

---

## Output Parser Types

### 1. StrOutputParser (Minimal)

**What:** Cleans up raw LLM output (removes whitespace, formatting).

```python
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()

text = "\n\n  The answer is: 42  \n\n"
cleaned = parser.parse(text)
# Output: "The answer is: 42"
```

**When to use:** Simple text extraction, no structure needed.

---

### 2. JSONOutputParser (Flexible)

**What:** Parses JSON responses, returns Python dict.

```python
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel

class BookReview(BaseModel):
    title: str
    rating: int
    summary: str

parser = JsonOutputParser(pydantic_object=BookReview)

# LLM returns JSON string
response = '{"title": "Python Guide", "rating": 5, "summary": "Excellent..."}'

review = parser.parse(response)
# Result: BookReview(title="...", rating=5, ...)
```

**Advantages:**
- Flexible (dict structure)
- Compatible with APIs
- Easy conversion to database

**Disadvantages:**
- No validation (any dict works)
- No type safety

---

### 3. PydanticOutputParser (Type-Safe, Strict)

**What:** Validates response against Pydantic model schema.

```python
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class ProductReview(BaseModel):
    product_name: str = Field(description="Name of product reviewed")
    rating: int = Field(ge=1, le=5, description="Rating 1-5")
    pros: list[str] = Field(description="List of positive aspects")
    cons: list[str] = Field(description="List of negative aspects")
    would_recommend: bool = Field(description="Whether to recommend")

parser = PydanticOutputParser(pydantic_object=ProductReview)

# Use in prompt
from langchain.prompts import PromptTemplate

template = """Extract review information:
{format_instructions}

Review: {review_text}"""

prompt = PromptTemplate(
    template=template,
    input_variables=["review_text"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# Chain it
from langchain_core.output_parsers import StrOutputParser as SP
from langchain_openai import ChatOpenAI

chain = prompt | ChatOpenAI() | parser

result = chain.invoke({
    "review_text": "Samsung phone is amazing! Camera quality is stunning..."
})

# Result:
# ProductReview(
#     product_name="Samsung phone",
#     rating=5,
#     pros=["Camera quality", "Performance"],
#     cons=[],
#     would_recommend=True
# )

# Type-safe access:
print(result.product_name)  # "Samsung phone"
print(result.rating)        # 5
print(isinstance(result, ProductReview))  # True
```

**Advantages:**
- Type safety
- Validation (constraints, ranges)
- Automatic conversion
- Pydantic error messages

**When to use:** Production systems, structured APIs, data extraction.

---

### 4. with_structured_output() (Modern, Recommended)

**What:** Built-in method on newer LLM models for structured outputs.

```python
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

class ProductReview(BaseModel):
    product_name: str = Field(description="Name of product")
    rating: int = Field(ge=1, le=5, description="Rating 1-5")
    summary: str = Field(description="Brief summary")
    recommendation: str = Field(enum=["buy", "skip"], description="Recommendation")

llm = ChatOpenAI(model="gpt-4")

# Bind structure directly to model
structured_llm = llm.with_structured_output(ProductReview)

# Now invoke directly with structured guarantee
review = structured_llm.invoke("""
Review the Samsung Galaxy S24 Ultra:
Amazing processor, stunning camera with 200MP and 100x zoom.
Battery lasts all day with 45W charging. Weighty for one-handed use.
Price is $1,300 which is expensive.
""")

# Result: ProductReview(product_name="...", rating=4, ...)
print(review.product_name)
print(review.rating)
```

**Advantages:**
- No need for separate parser
- Built into LLM (more reliable)
- Cleaner code
- Better error handling

**When to use:** Always prefer this for LLMs that support it (GPT-4, Claude 3+, Gemini).

---

## Pydantic Schema Design

### Basic Schema
```python
from pydantic import BaseModel, Field

class PersonProfile(BaseModel):
    name: str  # Required
    age: int   # Required
    email: str  # Required
```

### With Descriptions (For LLM)
```python
from pydantic import BaseModel, Field

class PersonProfile(BaseModel):
    name: str = Field(description="Full name of the person")
    age: int = Field(description="Age in years")
    email: str = Field(description="Email address for contact")
```

### With Constraints
```python
from pydantic import BaseModel, Field

class ProductRating(BaseModel):
    rating: int = Field(ge=1, le=5, description="Rating 1-5")
    text: str = Field(max_length=500, description="Review text ≤500 chars")
    helpful_count: int = Field(ge=0, description="Non-negative helpful count")
```

### With Optional Fields
```python
from typing import Optional
from pydantic import BaseModel, Field

class BookReview(BaseModel):
    title: str = Field(description="Book title")
    rating: int = Field(ge=1, le=5)
    review: str = Field(description="Detailed review")
    author_comment: Optional[str] = Field(default=None, description="Optional author response")
```

### With Defaults
```python
from pydantic import BaseModel, Field

class TaskStatus(BaseModel):
    task_id: str
    status: str = Field(default="pending", description="Task status")
    priority: int = Field(default=5, ge=1, le=10, description="1=low, 10=high")
    completed: bool = Field(default=False)
```

### With Enum (Fixed Values)
```python
from enum import Enum
from pydantic import BaseModel

class SentimentEnum(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"

class ReviewAnalysis(BaseModel):
    sentiment: SentimentEnum  # Only accepts these three values
    confidence: float  # 0-1
```

### Nested Models (Complex Structures)
```python
from typing import List
from pydantic import BaseModel

class LineItem(BaseModel):
    product_id: str
    quantity: int
    price: float

class Invoice(BaseModel):
    invoice_id: str
    items: List[LineItem]  # Nested list
    total: float
    customer_email: str
```

### List of Complex Objects
```python
from typing import List
from pydantic import BaseModel, Field

class Answer(BaseModel):
    question: str
    answer: str
    confidence: float = Field(ge=0, le=1)

class QAResult(BaseModel):
    topic: str
    qa_pairs: List[Answer]  # Multiple complex objects
```

### Example: Full Schema
```python
from typing import Optional, List
from enum import Enum
from pydantic import BaseModel, Field

class PriorityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class Subtask(BaseModel):
    name: str
    completed: bool = Field(default=False)

class Task(BaseModel):
    title: str = Field(description="Task title")
    description: str = Field(description="Detailed description")
    priority: PriorityLevel = Field(default=PriorityLevel.MEDIUM)
    assigned_to: str = Field(description="Assignee name")
    due_date: Optional[str] = Field(default=None, description="Due date (YYYY-MM-DD)")
    subtasks: List[Subtask] = Field(default_factory=list, description="List of subtasks")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    estimated_hours: float = Field(ge=0, description="Estimated effort in hours")

# Usage:
task = Task(
    title="Build API",
    description="Create REST API for users",
    priority=PriorityLevel.HIGH,
    assigned_to="Alice",
    due_date="2024-04-15",
    subtasks=[
        Subtask(name="Design endpoint", completed=True),
        Subtask(name="Implement endpoints"),
        Subtask(name="Add tests")
    ],
    tags=["backend", "api"],
    estimated_hours=8.5
)
```

---

## Implementation Patterns

### Pattern 1: Simple Chain with Parsing
```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

class SentimentResult(BaseModel):
    sentiment: str
    confidence: float

llm = ChatOpenAI()
structured_llm = llm.with_structured_output(SentimentResult)

template = "Analyze sentiment: {text}"
prompt = ChatPromptTemplate.from_template(template)

chain = prompt | structured_llm

result = chain.invoke({"text": "I love this product!"})
# Result: SentimentResult(sentiment="positive", confidence=0.95)
```

### Pattern 2: Multi-Step Extraction
```python
from typing import List
from pydantic import BaseModel

class Contact(BaseModel):
    name: str
    email: str
    phone: Optional[str]

class ContactExtraction(BaseModel):
    contacts: List[Contact]
    total_extracted: int

structured_llm = llm.with_structured_output(ContactExtraction)

contacts = structured_llm.invoke("""
Extract all contacts from this text:
John Doe (john@email.com, 555-1234)
Jane Smith (jane@company.com)
Bob Brown (bob@mail.com)
""")

# Result: ContactExtraction(
#     contacts=[
#         Contact(name="John Doe", email="john@email.com", phone="555-1234"),
#         ...
#     ],
#     total_extracted=3
# )
```

### Pattern 3: Chain with Custom Processing
```python
from langchain_core.runnables import RunnableLambda

class SentimentAnalysis(BaseModel):
    sentiment: str
    score: float

# Step 1: Get structured output
chain = prompt | llm.with_structured_output(SentimentAnalysis)

# Step 2: Custom processing
def process_sentiment(analysis: SentimentAnalysis) -> dict:
    return {
        "sentiment": analysis.sentiment,
        "score": analysis.score,
        "emoji": "😊" if analysis.sentiment == "positive" else "😞",
        "threshold_met": analysis.score > 0.7
    }

final_chain = chain | RunnableLambda(process_sentiment)

result = final_chain.invoke({"text": "Great service!"})
# Result: {"sentiment": "positive", "score": 0.92, "emoji": "😊", "threshold_met": True}
```

### Pattern 4: Batch Processing
```python
reviews = [
    "Love it!",
    "Terrible product",
    "It's okay"
]

structured_llm = llm.with_structured_output(SentimentResult)

# Batch process
results = structured_llm.batch(
    [{"text": review} for review in reviews]
)

for review, result in zip(reviews, results):
    print(f"{review} → {result.sentiment}")
```

### Pattern 5: Streaming Structured Output
```python
structured_llm = llm.with_structured_output(SentimentResult)

# Stream events
for event in structured_llm.stream({"text": "Great product honestly"}):
    print(event)
    # Output includes thinking process, then final structured result
```

---

## Error Recovery

### Problem 1: LLM Returns Invalid JSON

```python
# ❌ Problem
try:
    result = structured_llm.invoke(prompt)
except ValueError as e:
    # LLM returned invalid format, what now?
    print(f"Error: {e}")

# ✅ Solution: Use with_fallbacks
from langchain_core.runnables import RunnableWithFallbacks

fallback_llm = ChatOpenAI(model="gpt-3.5-turbo")  # Cheaper backup
fallback_structured = fallback_llm.with_structured_output(SentimentResult)

chain_with_fallback = structured_llm.with_fallbacks(
    [fallback_structured],
    exception_key="ValidationError"
)

result = chain_with_fallback.invoke(prompt)  # Tries GPT-4, falls back to 3.5-turbo
```

### Problem 2: Missing Required Fields

```python
# ❌ Problem
response = llm.invoke("Analyze: ")  # No actual text
# May return incomplete schema

# ✅ Solution: Validate before parsing
from pydantic import ValidationError

try:
    result = SentimentResult(**response_dict)
except ValidationError as e:
    # Handle missing/invalid fields
    print(f"Missing fields: {e.errors()}")
    # Either retry with clearer prompt or use defaults
    result = SentimentResult(
        sentiment="unknown",
        confidence=0.0
    )
```

### Problem 3: Constraint Violations

```python
# ❌ Problem: Rating outside range
class Review(BaseModel):
    rating: int = Field(ge=1, le=5)

# LLM might return rating=10 (wrong!)
try:
    review = Review(rating=10)
except ValidationError:
    print("Rating must be 1-5")

# ✅ Solution: Use stricter prompt + retry
from tenacity import retry, stop_after_attempt

@retry(stop=stop_after_attempt(3))
def parse_with_retry(text: str) -> Review:
    structured_llm = llm.with_structured_output(Review)
    return structured_llm.invoke({
        "text": text,
        "instructions": "Rating must be between 1 and 5 only"
    })

review = parse_with_retry(review_text)
```

### Problem 4: Enum Validation

```python
# ❌ Problem: LLM returns unexpected enum value
class Status(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"

result = Status("pending")  # ValueError!

# ✅ Solution: Map to closest valid value
def validate_status(value: str) -> Status:
    value_lower = value.lower()
    
    # Try direct match
    for status in Status:
        if status.value == value_lower:
            return status
    
    # Try fuzzy match
    from difflib import get_close_matches
    matches = get_close_matches(value_lower, [s.value for s in Status], n=1)
    if matches:
        return Status(matches[0])
    
    # Fallback
    return Status.ACTIVE

status = validate_status(llm_output)
```

---

## Real-World Examples

### Example 1: Invoice Extraction

```python
from typing import List
from pydantic import BaseModel, Field

class InvoiceLineItem(BaseModel):
    description: str
    quantity: int = Field(ge=1)
    unit_price: float = Field(ge=0)
    
    @property
    def total(self) -> float:
        return self.quantity * self.unit_price

class Invoice(BaseModel):
    invoice_number: str
    vendor: str
    date: str
    items: List[InvoiceLineItem]
    subtotal: float
    tax_rate: float = Field(ge=0, le=1)
    total: float
    
    def validate_total(self):
        expected = sum(item.total for item in self.items) * (1 + self.tax_rate)
        assert abs(expected - self.total) < 0.01, "Total mismatch"

# Usage
llm = ChatOpenAI()
invoice_parser = llm.with_structured_output(Invoice)

invoice = invoice_parser.invoke("""
Extract invoice details:
Invoice #INV-001
Vendor: TechCorp
Date: 2024-03-15
Items:
- Laptop x2 @ $999.99 = $1999.98
- Mouse x5 @ $29.99 = $149.95
Subtotal: $2149.93
Tax (10%): $214.99
Total: $2364.92
""")

print(f"Invoice for {invoice.vendor}: ${invoice.total}")
invoice.validate_total()  # Verify
```

### Example 2: Customer Feedback Analysis

```python
from typing import List
from enum import Enum
from pydantic import BaseModel, Field

class FeedbackCategory(str, Enum):
    PRODUCT_QUALITY = "product_quality"
    DELIVERY = "delivery"
    CUSTOMER_SERVICE = "service"
    PRICING = "pricing"

class FeedbackPoint(BaseModel):
    category: FeedbackCategory
    sentiment: str = Field(enum=["positive", "negative"])
    description: str

class CustomerFeedback(BaseModel):
    rating: int = Field(ge=1, le=5)
    feedback_points: List[FeedbackPoint]
    actionable_insights: List[str]
    recommend: bool

# Usage
feedback_parser = llm.with_structured_output(CustomerFeedback)

feedback = feedback_parser.invoke("""
Customer says: Product is excellent, arrived quickly.
However, pricing is a bit high compared to competitors.
Customer service was helpful when I had questions.
Overall very satisfied!
""")

# Now can analyze programmatically:
positive_count = sum(1 for p in feedback.feedback_points if p.sentiment == "positive")
print(f"Positive feedback points: {positive_count}")

for insight in feedback.actionable_insights:
    print(f"Action item: {insight}")
```

### Example 3: Meeting Transcript Summarization

```python
from typing import Optional
from pydantic import BaseModel

class ActionItem(BaseModel):
    task: str
    owner: str
    due_date: Optional[str]

class MeetingSummary(BaseModel):
    attendees: list[str]
    key_decisions: list[str]
    action_items: list[ActionItem]
    next_meeting: Optional[str]

# Usage
summary_parser = llm.with_structured_output(MeetingSummary)

summary = summary_parser.invoke("""
[Meeting Transcript]
Attendees: Alice, Bob, Charlie
Alice: We decided to launch v2 next quarter
Bob: I'll handle API development, Charlie handles frontend
Charlie: We need design review by next Friday
Alice: Next meeting is April 1st...
""")

# Structured access
for item in summary.action_items:
    print(f"{item.owner}: {item.task} ({item.due_date})")
```

---

## Interview Questions

**Q1: Explain structured output and why it matters.**
> LLMs naturally return unstructured text. Structured output converts this to typed, validated data (JSON, Pydantic models) that applications can process reliably. Essential for APIs, databases, and downstream systems.

**Q2: Compare PydanticOutputParser vs with_structured_output().**
> **PydanticOutputParser:** Manual parsing, separate from LLM, works with any model.
> **with_structured_output():** Built into model API, automatic, more reliable.
> Prefer with_structured_output() when available (GPT-4, Claude 3+, newer models).

**Q3: What constraints can Pydantic enforce?**
> - Type validation (int, str, bool, list, dict)
> - Range validation (Field(ge=1, le=5))
> - Length validation (max_length=500)
> - Enum validation (fixed set of values)
> - Custom validators (your own logic)
> - Nested models (complex structures)

**Q4: How do you handle LLM errors in structured output?**
> - Use `.with_fallbacks()` for retry logic
> - Implement `.ValidationError` exception handling
> - Retry with clearer prompts specifying constraints
> - Provide fallback values for missing fields
> - Use tenacity library for exponential backoff

**Q5: Design a Pydantic schema for a job listing.**
> ```python
> class JobListing(BaseModel):
>     title: str
>     company: str
>     salary_min: float = Field(ge=0)
>     salary_max: float = Field(ge=0)
>     experience_level: Literal["entry", "mid", "senior"]
>     location: str
>     remote: bool = Field(default=False)
>     skills_required: List[str]
> ```

**Q6: Explain the difference between StrOutputParser and JsonOutputParser.**
> **StrOutputParser:** Cleans text, returns string.
> **JsonOutputParser:** Parses JSON, returns dict or Pydantic model.
> Choose StrOutputParser for text extraction; JsonOutputParser for structured data.

**Q7: What's the benefit of nested Pydantic models?**
> Enables complex, hierarchical structures:
> ```python
> class Address(BaseModel):
>     street: str
>     city: str
> 
> class Person(BaseModel):
>     name: str
>     address: Address  # Nested!
> ```
> LLM returns hierarchically-structured data matching real-world relationships.

**Q8: How would you validate an extracted list of emails?**
> ```python
> from pydantic import EmailStr, validator
> from typing import List
> 
> class EmailList(BaseModel):
>     emails: List[EmailStr]
>     
>     @validator("emails")
>     def no_duplicates(cls, v):
>         if len(v) != len(set(v)):
>             raise ValueError("Duplicate emails found")
>         return v
> ```

**Q9: Design a schema for movie review extraction.**
> ```python
> class MovieReview(BaseModel):
>     title: str
>     director: str
>     rating: float = Field(ge=0, le=10)
>     genres: List[str]
>     plot_summary: str = Field(max_length=500)
>     recommendation: Literal["highly_recommend", "recommend", "skip"]
> ```

**Q10: What happens if LLM returns data outside constraints?**
> Pydantic raises ValidationError with detailed error info. Options:
> 1. Retry LLM with stricter prompt
> 2. Use fallback model
> 3. Apply constraint correction (clamp to range)
> 4. Require user correction
> Always prefer prevention (clear prompt) over recovery.

---

**Key Takeaway:** Structured output transforms raw LLM responses into usable, validated data. Use Pydantic schemas for clear constraints, prefer with_structured_output() for modern models, and implement robust error handling for production systems.
