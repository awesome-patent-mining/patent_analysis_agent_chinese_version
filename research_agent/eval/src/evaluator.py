from jinja2 import Environment, FileSystemLoader

from .llm import LLM


class Evaluator:
    """A class to evaluate survey papers using LLM-based analysis.

    This class handles loading prompt templates and evaluating survey papers
    using a language model.

    Attributes:
        prompt_dir (str): Directory containing prompt templates
        prompt_file (str): Name of the template file to use
        prompt_template: Loaded Jinja template
        llm: Language model instance
    """

    def __init__(
        self, prompt_dir: str = "eval/prompts", prompt_file: str = "evaluation.jinja"
    ):
        """Initialize the Evaluator.

        Args:
            prompt_dir (str): Directory containing prompt templates
            prompt_file (str): Name of the template file to use

        Raises:
            FileNotFoundError: If prompt directory or file doesn't exist
        """
        self.prompt_dir = prompt_dir
        self.prompt_file = prompt_file
        self.prompt_template = self._load_prompt()
        self.llm = LLM()

    def _load_prompt(self):
        """Load the Jinja template from the specified directory.

        Returns:
            Template: Loaded Jinja template

        Raises:
            FileNotFoundError: If template file cannot be found
            jinja2.TemplateError: If template is invalid
        """
        try:
            env = Environment(loader=FileSystemLoader(self.prompt_dir))
            template = env.get_template(self.prompt_file)
            return template
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Prompt directory or file not found: {self.prompt_dir}/{self.prompt_file}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load prompt template: {str(e)}")

    async def evaluate(self, survey_paper: str):
        """Evaluate a survey paper using the loaded prompt template.

        Args:
            survey_paper (str): Content of the survey paper to evaluate

        Returns:
            str: Evaluation results from the language model

        Raises:
            ValueError: If survey_paper is empty or invalid
            RuntimeError: If LLM completion fails
        """
        if not survey_paper or not isinstance(survey_paper, str):
            raise ValueError("Survey paper must be a non-empty string")

        try:
            system_prompt = self.prompt_template.render(role="system")
            user_prompt = self.prompt_template.render(
                role="user", survey_paper=survey_paper
            )
            prompt = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            evaluation = await self.llm.completion(prompt)
            return evaluation
        except Exception as e:
            raise RuntimeError(f"Evaluation failed: {str(e)}")
