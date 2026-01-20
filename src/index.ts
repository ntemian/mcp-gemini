#!/usr/bin/env node

/**
 * Gemini MCP Server
 * Provides Claude Code access to Google Gemini API
 *
 * Models available:
 * - gemini-2.0-flash-exp: Latest experimental (fast, multimodal)
 * - gemini-1.5-pro: Best for complex tasks
 * - gemini-1.5-flash: Fast and efficient
 * - gemini-1.5-flash-8b: Smallest, fastest
 */

import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { z } from 'zod';
import { GoogleGenerativeAI, HarmCategory, HarmBlockThreshold } from '@google/generative-ai';

// Initialize Gemini client
const genAI = new GoogleGenerativeAI(process.env.GOOGLE_AI_API_KEY || '');

// Safety settings (permissive for general use)
const safetySettings = [
  { category: HarmCategory.HARM_CATEGORY_HARASSMENT, threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH },
  { category: HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH },
  { category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH },
  { category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH },
];

// Create MCP server
const server = new McpServer({
  name: 'gemini',
  version: '1.0.0',
});

// ============================================
// TYPES
// ============================================

interface ChatMessage {
  role: 'user' | 'model';
  parts: { text: string }[];
}

// ============================================
// CHAT TOOLS
// ============================================

server.tool(
  'gemini_chat',
  'Send a chat completion to Google Gemini. Strong multimodal and reasoning capabilities.',
  {
    messages: z.array(z.object({
      role: z.enum(['user', 'model']),
      content: z.string(),
    })).describe('Array of messages (role: user or model)'),
    model: z.string().optional().describe('Model: gemini-2.0-flash-exp (default), gemini-1.5-pro, gemini-1.5-flash'),
    temperature: z.number().optional().describe('Sampling temperature 0-2. Default: 1'),
    max_tokens: z.number().optional().describe('Maximum tokens to generate'),
  },
  async (params) => {
    try {
      const { messages, model, temperature, max_tokens } = params;
      const geminiModel = genAI.getGenerativeModel({
        model: model || 'gemini-2.0-flash-exp',
        safetySettings,
        generationConfig: {
          temperature: temperature ?? 1,
          maxOutputTokens: max_tokens,
        },
      });

      // Convert to Gemini format
      const history: ChatMessage[] = messages.slice(0, -1).map(m => ({
        role: m.role === 'user' ? 'user' : 'model',
        parts: [{ text: m.content }],
      }));

      const lastMessage = messages[messages.length - 1];
      const chat = geminiModel.startChat({ history });
      const result = await chat.sendMessage(lastMessage.content);
      const response = result.response;

      return {
        content: [{
          type: 'text' as const,
          text: response.text(),
        }],
      };
    } catch (error: unknown) {
      const err = error as { message: string };
      return {
        content: [{ type: 'text' as const, text: `Gemini API error: ${err.message}` }],
        isError: true,
      };
    }
  }
);

server.tool(
  'gemini_complete',
  'Simple one-shot completion with Gemini.',
  {
    prompt: z.string().describe('The prompt to send'),
    system: z.string().optional().describe('Optional system instruction'),
    model: z.string().optional().describe('Model to use. Default: gemini-2.0-flash-exp'),
    temperature: z.number().optional().describe('Temperature 0-2'),
  },
  async (params) => {
    try {
      const { prompt, system, model, temperature } = params;
      const geminiModel = genAI.getGenerativeModel({
        model: model || 'gemini-2.0-flash-exp',
        systemInstruction: system,
        safetySettings,
        generationConfig: {
          temperature: temperature ?? 1,
        },
      });

      const result = await geminiModel.generateContent(prompt);
      return {
        content: [{ type: 'text' as const, text: result.response.text() }],
      };
    } catch (error: unknown) {
      const err = error as { message: string };
      return {
        content: [{ type: 'text' as const, text: `Gemini API error: ${err.message}` }],
        isError: true,
      };
    }
  }
);

// ============================================
// MODEL-SPECIFIC SHORTCUTS
// ============================================

server.tool(
  'gemini_pro',
  'Use Gemini 1.5 Pro for complex tasks requiring deep reasoning.',
  {
    prompt: z.string().describe('The prompt'),
    system: z.string().optional().describe('Optional system instruction'),
    temperature: z.number().optional().describe('Temperature 0-2'),
  },
  async (params) => {
    try {
      const { prompt, system, temperature } = params;
      const geminiModel = genAI.getGenerativeModel({
        model: 'gemini-1.5-pro',
        systemInstruction: system,
        safetySettings,
        generationConfig: { temperature: temperature ?? 1 },
      });

      const result = await geminiModel.generateContent(prompt);
      return {
        content: [{ type: 'text' as const, text: result.response.text() }],
      };
    } catch (error: unknown) {
      const err = error as { message: string };
      return {
        content: [{ type: 'text' as const, text: `Gemini Pro error: ${err.message}` }],
        isError: true,
      };
    }
  }
);

server.tool(
  'gemini_flash',
  'Use Gemini Flash for fast, efficient responses.',
  {
    prompt: z.string().describe('The prompt'),
    system: z.string().optional().describe('Optional system instruction'),
  },
  async (params) => {
    try {
      const { prompt, system } = params;
      const geminiModel = genAI.getGenerativeModel({
        model: 'gemini-1.5-flash',
        systemInstruction: system,
        safetySettings,
      });

      const result = await geminiModel.generateContent(prompt);
      return {
        content: [{ type: 'text' as const, text: result.response.text() }],
      };
    } catch (error: unknown) {
      const err = error as { message: string };
      return {
        content: [{ type: 'text' as const, text: `Gemini Flash error: ${err.message}` }],
        isError: true,
      };
    }
  }
);

// ============================================
// CODE TOOLS
// ============================================

server.tool(
  'gemini_code',
  'Generate code using Gemini. Strong at multiple languages.',
  {
    task: z.string().describe('Description of what code to generate'),
    language: z.string().optional().describe('Programming language'),
    context: z.string().optional().describe('Additional context or existing code'),
  },
  async (params) => {
    try {
      const { task, language, context } = params;
      const systemPrompt = `You are an expert programmer. Generate clean, efficient, well-documented code.${language ? ` Use ${language}.` : ''} Only output the code with minimal explanation.`;

      let userPrompt = task;
      if (context) {
        userPrompt = `Context:\n${context}\n\nTask: ${task}`;
      }

      const geminiModel = genAI.getGenerativeModel({
        model: 'gemini-1.5-pro',
        systemInstruction: systemPrompt,
        safetySettings,
        generationConfig: { temperature: 0.2 },
      });

      const result = await geminiModel.generateContent(userPrompt);
      return {
        content: [{ type: 'text' as const, text: result.response.text() }],
      };
    } catch (error: unknown) {
      const err = error as { message: string };
      return {
        content: [{ type: 'text' as const, text: `Gemini error: ${err.message}` }],
        isError: true,
      };
    }
  }
);

server.tool(
  'gemini_analyze',
  'Analyze code using Gemini.',
  {
    code: z.string().describe('The code to analyze'),
    task: z.enum(['explain', 'review', 'bugs', 'improve', 'security']).describe('Type of analysis'),
  },
  async (params) => {
    try {
      const { code, task } = params;
      const taskPrompts: Record<string, string> = {
        explain: 'Explain what this code does in detail.',
        review: 'Review this code for quality, maintainability, and best practices.',
        bugs: 'Find any bugs, edge cases, or potential issues.',
        improve: 'Suggest improvements for better code quality.',
        security: 'Analyze for security vulnerabilities.',
      };

      const geminiModel = genAI.getGenerativeModel({
        model: 'gemini-1.5-pro',
        systemInstruction: 'You are an expert code analyst.',
        safetySettings,
        generationConfig: { temperature: 0.3 },
      });

      const result = await geminiModel.generateContent(
        `${taskPrompts[task]}\n\nCode:\n\`\`\`\n${code}\n\`\`\``
      );
      return {
        content: [{ type: 'text' as const, text: result.response.text() }],
      };
    } catch (error: unknown) {
      const err = error as { message: string };
      return {
        content: [{ type: 'text' as const, text: `Gemini error: ${err.message}` }],
        isError: true,
      };
    }
  }
);

// ============================================
// REASONING TOOL
// ============================================

server.tool(
  'gemini_reason',
  'Use Gemini for complex reasoning and problem-solving with step-by-step thinking.',
  {
    problem: z.string().describe('The problem or question requiring deep reasoning'),
    context: z.string().optional().describe('Additional context'),
  },
  async (params) => {
    try {
      const { problem, context } = params;

      let prompt = problem;
      if (context) {
        prompt = `Context:\n${context}\n\nProblem: ${problem}`;
      }

      const geminiModel = genAI.getGenerativeModel({
        model: 'gemini-1.5-pro',
        systemInstruction: 'Think step by step. Show your reasoning process clearly. Break down complex problems into smaller parts.',
        safetySettings,
        generationConfig: { temperature: 0.2 },
      });

      const result = await geminiModel.generateContent(prompt);
      return {
        content: [{ type: 'text' as const, text: result.response.text() }],
      };
    } catch (error: unknown) {
      const err = error as { message: string };
      return {
        content: [{ type: 'text' as const, text: `Gemini error: ${err.message}` }],
        isError: true,
      };
    }
  }
);

// ============================================
// UTILITY TOOLS
// ============================================

server.tool(
  'gemini_models',
  'List available Gemini models and their capabilities.',
  {},
  async () => {
    const models = `
Gemini Models:

1. gemini-2.0-flash-exp (Recommended)
   - Latest experimental model
   - Fast, multimodal (text, images, audio, video)
   - 1M token context window
   - Best for: Most tasks, speed + capability

2. gemini-1.5-pro
   - Most capable stable model
   - 2M token context window
   - Strong reasoning and coding
   - Best for: Complex tasks, long documents

3. gemini-1.5-flash
   - Fast and efficient
   - 1M token context window
   - Good balance of speed/quality
   - Best for: Quick responses, high volume

4. gemini-1.5-flash-8b
   - Smallest, fastest model
   - Lower cost
   - Best for: Simple tasks, cost-sensitive

Pricing (per 1M tokens, 1.5 Pro):
- Input: $1.25 (≤128K) / $2.50 (>128K)
- Output: $5.00 (≤128K) / $10.00 (>128K)

Features:
- Multimodal: Images, video, audio, code
- Long context: Up to 2M tokens
- Grounding: Google Search integration (API)
- Code execution: Built-in Python sandbox
`;

    return {
      content: [{ type: 'text' as const, text: models.trim() }],
    };
  }
);

server.tool(
  'gemini_summarize',
  'Summarize text using Gemini.',
  {
    text: z.string().describe('Text to summarize'),
    style: z.enum(['brief', 'detailed', 'bullets']).optional().describe('Summary style. Default: brief'),
  },
  async (params) => {
    try {
      const { text, style } = params;
      const stylePrompts: Record<string, string> = {
        brief: 'Provide a brief 2-3 sentence summary.',
        detailed: 'Provide a detailed summary covering all main points.',
        bullets: 'Summarize as bullet points.',
      };

      const geminiModel = genAI.getGenerativeModel({
        model: 'gemini-1.5-flash',
        safetySettings,
        generationConfig: { temperature: 0.3 },
      });

      const result = await geminiModel.generateContent(
        `${stylePrompts[style || 'brief']}\n\nText:\n${text}`
      );
      return {
        content: [{ type: 'text' as const, text: result.response.text() }],
      };
    } catch (error: unknown) {
      const err = error as { message: string };
      return {
        content: [{ type: 'text' as const, text: `Gemini error: ${err.message}` }],
        isError: true,
      };
    }
  }
);

server.tool(
  'gemini_translate',
  'Translate text using Gemini.',
  {
    text: z.string().describe('Text to translate'),
    target_language: z.string().describe('Target language (e.g., "English", "Greek", "Japanese")'),
    source_language: z.string().optional().describe('Source language (auto-detect if not specified)'),
  },
  async (params) => {
    try {
      const { text, target_language, source_language } = params;

      const prompt = source_language
        ? `Translate the following ${source_language} text to ${target_language}. Only output the translation.\n\nText: ${text}`
        : `Translate the following text to ${target_language}. Only output the translation.\n\nText: ${text}`;

      const geminiModel = genAI.getGenerativeModel({
        model: 'gemini-1.5-flash',
        safetySettings,
        generationConfig: { temperature: 0.3 },
      });

      const result = await geminiModel.generateContent(prompt);
      return {
        content: [{ type: 'text' as const, text: result.response.text() }],
      };
    } catch (error: unknown) {
      const err = error as { message: string };
      return {
        content: [{ type: 'text' as const, text: `Gemini error: ${err.message}` }],
        isError: true,
      };
    }
  }
);

// ============================================
// START SERVER
// ============================================

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error('Gemini MCP server running');
}

main().catch(console.error);
