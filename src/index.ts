#!/usr/bin/env node

/**
 * Gemini MCP Server
 * Provides Claude Code access to Google Gemini API
 *
 * Models available:
 * - gemini-3.1-pro-preview: Latest flagship (complex reasoning, coding)
 * - gemini-2.5-flash: Fast and efficient
 * - gemini-3-pro-preview: Previous gen pro
 */

import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { z } from 'zod';
import { GoogleGenerativeAI, HarmCategory, HarmBlockThreshold } from '@google/generative-ai';
import { GoogleGenAI, Scale, MusicGenerationMode } from '@google/genai';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';

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
    model: z.string().optional().describe('Model: gemini-2.5-flash (default), gemini-3.1-pro-preview, gemini-3-pro-preview'),
    temperature: z.number().optional().describe('Sampling temperature 0-2. Default: 1'),
    max_tokens: z.number().optional().describe('Maximum tokens to generate'),
  },
  async (params) => {
    try {
      const { messages, model, temperature, max_tokens } = params;
      const geminiModel = genAI.getGenerativeModel({
        model: model || 'gemini-2.5-flash',
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
    model: z.string().optional().describe('Model to use. Default: gemini-2.5-flash'),
    temperature: z.number().optional().describe('Temperature 0-2'),
  },
  async (params) => {
    try {
      const { prompt, system, model, temperature } = params;
      const geminiModel = genAI.getGenerativeModel({
        model: model || 'gemini-2.5-flash',
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
  'Use Gemini 3.1 Pro for complex tasks requiring deep reasoning. Latest and best model (released 2026-02-19).',
  {
    prompt: z.string().describe('The prompt'),
    system: z.string().optional().describe('Optional system instruction'),
    temperature: z.number().optional().describe('Temperature 0-2'),
  },
  async (params) => {
    try {
      const { prompt, system, temperature } = params;
      const geminiModel = genAI.getGenerativeModel({
        model: 'gemini-3.1-pro-preview',
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
  'Use Gemini 2.5 Flash for fast, efficient responses.',
  {
    prompt: z.string().describe('The prompt'),
    system: z.string().optional().describe('Optional system instruction'),
  },
  async (params) => {
    try {
      const { prompt, system } = params;
      const geminiModel = genAI.getGenerativeModel({
        model: 'gemini-2.5-flash',
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
        model: 'gemini-3.1-pro-preview',
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
        model: 'gemini-3.1-pro-preview',
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
        model: 'gemini-3.1-pro-preview',
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

1. gemini-3.1-pro-preview ★ NEW (2026-02-19)
   - Latest flagship model
   - ARC-AGI-2: 77.1%, GPQA Diamond: 94.3%, SWE-Bench: 80.6%
   - Deep reasoning, complex problem-solving
   - Best for: Complex tasks, coding, analysis

2. gemini-2.5-flash (Default)
   - Fast and efficient
   - Great balance of speed and quality
   - Best for: Most tasks, quick responses

3. gemini-3-pro-preview (Previous gen)
   - Previous flagship
   - Still strong for general tasks

Features:
- Multimodal: Images, video, audio, code
- Grounding: Google Search integration
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
        model: 'gemini-3.1-pro-preview',
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
        model: 'gemini-3.1-pro-preview',
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
// LYRIA MUSIC GENERATION (Google Lyria RealTime)
// ============================================

const genAIMusic = new GoogleGenAI({
  apiKey: process.env.GOOGLE_AI_API_KEY || '',
  httpOptions: { apiVersion: 'v1alpha' },
});

const LYRIA_MODEL = 'models/lyria-realtime-exp';

// Scale name mapping for user-friendly input
const SCALE_MAP: Record<string, Scale> = {
  'C': Scale.C_MAJOR_A_MINOR,
  'C major': Scale.C_MAJOR_A_MINOR,
  'Db': Scale.D_FLAT_MAJOR_B_FLAT_MINOR,
  'Db major': Scale.D_FLAT_MAJOR_B_FLAT_MINOR,
  'D': Scale.D_MAJOR_B_MINOR,
  'D major': Scale.D_MAJOR_B_MINOR,
  'Eb': Scale.E_FLAT_MAJOR_C_MINOR,
  'Eb major': Scale.E_FLAT_MAJOR_C_MINOR,
  'E': Scale.E_MAJOR_D_FLAT_MINOR,
  'E major': Scale.E_MAJOR_D_FLAT_MINOR,
  'F': Scale.F_MAJOR_D_MINOR,
  'F major': Scale.F_MAJOR_D_MINOR,
  'Gb': Scale.G_FLAT_MAJOR_E_FLAT_MINOR,
  'Gb major': Scale.G_FLAT_MAJOR_E_FLAT_MINOR,
  'G': Scale.G_MAJOR_E_MINOR,
  'G major': Scale.G_MAJOR_E_MINOR,
  'Ab': Scale.A_FLAT_MAJOR_F_MINOR,
  'Ab major': Scale.A_FLAT_MAJOR_F_MINOR,
  'A': Scale.A_MAJOR_G_FLAT_MINOR,
  'A major': Scale.A_MAJOR_G_FLAT_MINOR,
  'Bb': Scale.B_FLAT_MAJOR_G_MINOR,
  'Bb major': Scale.B_FLAT_MAJOR_G_MINOR,
  'B': Scale.B_MAJOR_A_FLAT_MINOR,
  'B major': Scale.B_MAJOR_A_FLAT_MINOR,
};

function writeWav(samples: Int16Array, filePath: string) {
  const numChannels = 2;
  const sampleRate = 48000;
  const bitsPerSample = 16;
  const byteRate = sampleRate * numChannels * (bitsPerSample / 8);
  const blockAlign = numChannels * (bitsPerSample / 8);
  const dataSize = samples.length * (bitsPerSample / 8);
  const headerSize = 44;

  const buffer = Buffer.alloc(headerSize + dataSize);

  // RIFF header
  buffer.write('RIFF', 0);
  buffer.writeUInt32LE(headerSize - 8 + dataSize, 4);
  buffer.write('WAVE', 8);

  // fmt chunk
  buffer.write('fmt ', 12);
  buffer.writeUInt32LE(16, 16); // chunk size
  buffer.writeUInt16LE(1, 20);  // PCM format
  buffer.writeUInt16LE(numChannels, 22);
  buffer.writeUInt32LE(sampleRate, 24);
  buffer.writeUInt32LE(byteRate, 28);
  buffer.writeUInt16LE(blockAlign, 32);
  buffer.writeUInt16LE(bitsPerSample, 34);

  // data chunk
  buffer.write('data', 36);
  buffer.writeUInt32LE(dataSize, 40);

  // Write samples
  for (let i = 0; i < samples.length; i++) {
    buffer.writeInt16LE(samples[i], headerSize + i * 2);
  }

  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, buffer);
}

server.tool(
  'gemini_music',
  'Generate instrumental music using Google Lyria RealTime. Streams audio via WebSocket, saves as WAV file. Free during preview. Instrumental only (no vocals).',
  {
    prompts: z.array(z.object({
      text: z.string().describe('Music description (genre, instruments, mood, style)'),
      weight: z.number().optional().describe('Influence weight (default: 1.0, higher = more influence)'),
    })).describe('Array of weighted music prompts'),
    duration_seconds: z.number().optional().describe('Duration in seconds (default: 30, max: 60)'),
    bpm: z.number().optional().describe('Beats per minute (60-200)'),
    density: z.number().optional().describe('Note density 0.0 (sparse) to 1.0 (dense)'),
    brightness: z.number().optional().describe('Tonal brightness 0.0 (dark) to 1.0 (bright)'),
    guidance: z.number().optional().describe('Prompt adherence 0.0 (free) to 6.0 (strict). Default: 4.0'),
    scale: z.string().optional().describe('Musical key: C, D, E, F, G, A, B, Db, Eb, Gb, Ab, Bb'),
    output_path: z.string().optional().describe('Output WAV path (default: ~/Downloads/lyria-{timestamp}.wav)'),
  },
  async (params) => {
    try {
      const { prompts, duration_seconds, bpm, density, brightness, guidance, scale, output_path } = params;
      const duration = Math.min(duration_seconds ?? 30, 60);

      const outputFile = output_path || path.join(
        os.homedir(), 'Downloads',
        `lyria-${new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)}.wav`
      );

      const audioChunks: number[][] = [];
      let chunkCount = 0;

      const session = await genAIMusic.live.music.connect({
        model: LYRIA_MODEL,
        callbacks: {
          onmessage: (message) => {
            const chunk = message.audioChunk;
            if (chunk?.data) {
              const audioBuffer = Buffer.from(chunk.data, 'base64');
              const intArray = new Int16Array(
                audioBuffer.buffer,
                audioBuffer.byteOffset,
                audioBuffer.length / Int16Array.BYTES_PER_ELEMENT
              );
              audioChunks.push(Array.from(intArray));
              chunkCount++;
            }
          },
          onerror: (error) => {
            console.error('Lyria session error:', error);
          },
          onclose: () => {
            console.error('Lyria session closed');
          },
        },
      });

      // Set prompts
      await session.setWeightedPrompts({
        weightedPrompts: prompts.map(p => ({
          text: p.text,
          weight: p.weight ?? 1.0,
        })),
      });

      // Set config
      const config: Record<string, unknown> = {};
      if (bpm !== undefined) config.bpm = bpm;
      if (density !== undefined) config.density = density;
      if (brightness !== undefined) config.brightness = brightness;
      if (guidance !== undefined) config.guidance = guidance;
      if (scale && SCALE_MAP[scale]) config.scale = SCALE_MAP[scale];

      if (Object.keys(config).length > 0) {
        await session.setMusicGenerationConfig({
          musicGenerationConfig: config,
        });
      }

      // Play and collect audio for the specified duration
      session.play();

      await new Promise(resolve => setTimeout(resolve, duration * 1000));

      session.close();

      // Wait a moment for any remaining chunks
      await new Promise(resolve => setTimeout(resolve, 500));

      if (audioChunks.length === 0) {
        return {
          content: [{ type: 'text' as const, text: 'No audio received from Lyria. The model may be unavailable or rate-limited.' }],
          isError: true,
        };
      }

      // Combine all chunks and write WAV
      const allSamples = new Int16Array(audioChunks.flat());
      writeWav(allSamples, outputFile);

      const durationActual = (allSamples.length / 2 / 48000).toFixed(1); // stereo
      const fileSizeMB = (fs.statSync(outputFile).size / 1024 / 1024).toFixed(1);

      return {
        content: [{
          type: 'text' as const,
          text: `Music generated successfully!\n\nFile: ${outputFile}\nDuration: ${durationActual}s\nSize: ${fileSizeMB} MB\nFormat: WAV 48kHz 16-bit stereo\nChunks received: ${chunkCount}\n\nPrompts: ${prompts.map(p => `"${p.text}" (weight: ${p.weight ?? 1.0})`).join(', ')}`,
        }],
      };
    } catch (error: unknown) {
      const err = error as { message: string };
      return {
        content: [{ type: 'text' as const, text: `Lyria music error: ${err.message}` }],
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
