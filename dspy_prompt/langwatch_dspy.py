from typing import Dict

import dspy
import langwatch
import toml
from dspy import evaluate
from dspy.datasets import HotPotQA
from dspy.teleprompt import BootstrapFewShot

model_name = 'llama3'
lm = dspy.OllamaLocal(model=model_name)

colbertv2_wiki17_abstracts = dspy.ColBERTv2(
    url='http://20.102.90.50:2017/wiki17_abstracts')

dspy.settings.configure(lm=lm, rm=colbertv2_wiki17_abstracts)


class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""
    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")


class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()

        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)


def load_dataset():
    dataset = HotPotQA(train_seed=1, train_size=20, eval_seed=2023, dev_size=50,
                       test_size=0)

    return [x.with_inputs('question') for x in dataset.train]


def validate_context_and_answer(example, prediction, trace=None):
    return evaluate.answer_exact_match(example, prediction) \
        and evaluate.answer_passage_match(example, prediction)


def load_toml() -> Dict[str, str]:
    with open('dspy_prompt/pyproject.toml', 'r') as f:
        data = toml.load(f)
        config = data.get("langwatch").get("config")
    return config


if __name__ == '__main__':
    cfg = load_toml()

    langwatch.endpoint = cfg["endpoint"]
    langwatch.api_key = cfg["api_key"]

    train_set = load_dataset()

    teleprompter = BootstrapFewShot(metric=validate_context_and_answer)
    langwatch.dspy.init(experiment="rag-dspy-tutorial", optimizer=teleprompter)

    compiled_rag = teleprompter.compile(RAG(), trainset=train_set)
