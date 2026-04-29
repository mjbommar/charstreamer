# Security

Do not commit API keys, credentials, generated LLM outputs containing sensitive
source text, local logs, or model artifacts.

The span-generator reads `OPENAI_API_KEY` from the process environment or from a
local `--env-json` file. Local environment files are ignored by git.

To report a security issue, use a private channel to contact the maintainer
rather than filing a public issue with exploit details.
