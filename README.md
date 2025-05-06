# LLM_Chatbot_Arena
Abstract:

Evaluating chatbot responses is a critical challenge in conversational AI. While
human evaluation is the gold standard, it is costly and time-consuming. Re-
cent advances in LLMs, such as GPT-4, have enabled automated evaluation,
but the degree to which these models align with human judgment—especially
across different question types—remains unclear. This project investigates the
alignment between automated GPT-based judgment and human judgment in
evaluating chatbot responses. By comparing GPT-4 and human preferences
on both open-ended and close-ended questions, we aim to identify where mis-
alignments are most pronounced. Additionally, we train a predictive model to
classify user prompts into one of five semantic domains encapsulating both open
and close ended domains and then predict alignment patterns between human
and GPT-based evaluations Our findings provide insights into the strengths and
limitations of LLM-based evaluation for conversational AI.

To take our project a step further and make it interactive, we developed a
web-based interactive dashboard using the Plotly Dash. This application allows
the user to enter any chatbot prompt and instantly receive a predicted domain
classification (math, code, factual, creative, or opinion) based on our trained
best svm oc model.
The dashboard was built using Python and Dash, and integrates our final
SVM model, TF-IDF vectorizer, and label encoder. It features:
• A clean user interface with a text input box and prediction display
• Real-time inference using the SVM classifier trained on prompt + prompt
type
• Visualizations showing domain distribution and model win counts from
the dataset
