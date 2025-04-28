from pathlib import Path

# 获取core目录的路径
CORE_DIR = Path(__file__).parent

# prompts目录路径
PROMPTS_DIR = CORE_DIR / "prompts"

# 各个prompt文件路径
ANSWER_QUESTION_PROMPT = PROMPTS_DIR / "answer_question.jinja"
PAPER_REVIEW_PROMPT = PROMPTS_DIR / "paper_review.jinja"
CREATE_QUESTIONS_PROMPT = PROMPTS_DIR / "create_questions.jinja"
INTRODUCTION_PROMPT = PROMPTS_DIR / "introduction_writer.jinja"
RELATED_WORK_PROMPT = PROMPTS_DIR / "related_work_writer.jinja"
CONCLUSION_PROMPT = PROMPTS_DIR / "conclusion_writer.jinja"
FIND_STATEMENT_CITATION_PROMPT = PROMPTS_DIR / "find_statement_citation.jinja"
ADD_CITATIONS_PROMPT = PROMPTS_DIR / "add_citations.jinja"
VERIFY_STATEMENT_CITATION_PROMPT = PROMPTS_DIR / "verify_statement_citation.jinja"
UPDATE_REFERENCE_PROMPT = PROMPTS_DIR / "update_reference.jinja"
HYDE_PROMPT = PROMPTS_DIR / "hyde_prompt.jinja"
REWRITE_OUTLINE_PROMPT = PROMPTS_DIR / "rewrite_outline.jinja"
REWRITE_SECTION_PROMPT = PROMPTS_DIR / "rewrite_section.jinja"
WRITE_SECTION_PROMPT = PROMPTS_DIR / "write_section.jinja"
GENE_HYDE_PROMPT = PROMPTS_DIR / "gene_hyde.jinja"
