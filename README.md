# LLM Randomness Analysis

An empirical analysis of random number generation capabilities in Large Language Models (LLMs) compared to traditional random number generators.

## Project Overview

This project investigates whether AI language models can produce truly random numbers when prompted. Through statistical analysis and visualization, we compare the randomness characteristics of various popular closed-source and open-source models against traditional pseudo-random number generators.

## Key Findings

- AI language models exhibit varying degrees of deviation from true randomness
- Traditional random number generators produce more uniformly distributed numbers
- Some AI models show biases toward specific values within given ranges
- Range size influences the randomness of AI model outputs
- AI models show notable deviations in generating prime numbers and round numbers

## Models Tested

### AI Language Models
- chatgpt-4o-latest (tested date: 2024-11-02)
- gpt-4-0613
- gpt-4o-2024-08-06
- gpt-4o-mini-2024-07-18
- llama-3.2-3b-instruct
- gemma-2-9b-it
- gemma-2-2b-it

### Traditional Random Number Generators
- Python's built-in random module
- NumPy's random generator

## Methodology

### Data Collection
- Generated 200 random numbers for each model within specified ranges
- Ranges tested:
  - 0 to 10
  - 0 to 100
  - -745 to -556
  - -359 to 39
  - 328 to 508
  - -647 to 489
  - -863 to 557

### Analysis Components
1. Data Collection and Validation
2. Response Normalization
3. Statistical Analysis
4. Visualization
5. Pattern Recognition

## Results

The analysis includes several key visualizations:
- Distribution of Random Values by Range
- Mean Normalized Response vs. Range Size
- Distribution of Prime Numbers
- Distribution of Round Numbers
- Boxplot of Normalized Responses

## Conclusions

While AI language models excel at generating human-like text, their ability to produce truly random numbers is limited. The study reveals inherent biases in AI-generated numbers, suggesting that traditional random number generators remain the more reliable choice for applications requiring true randomness.

## Usage

To replicate the results of this project for yourself, set your OpenAI API key in the `generate_numbers_llms.py` script and run the script. A CSV file will be generated in the `data/runs` directory.

To generate results for local model (I used LM-Studio), run the `generate_numbers_llms_local.py` script and a similar CSV file will be generated in the `data/runs` directory.

## Requirements

See `requirements.txt` for a list of dependencies. Install dependencies with `pip install -r requirements.txt`.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Further Reading

For a detailed analysis of this project, check out the [full article on Medium](https://medium.com/@kmaurinjones/evaluating-randomness-in-generative-ai-large-language-models-099f747b28e2).

## Author

**Kai Maurin-Jones**
[LinkedIn](https://www.linkedin.com/in/kmaurinjones/)

## License

MIT License

Copyright (c) 2024 Kai Maurin-Jones

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
