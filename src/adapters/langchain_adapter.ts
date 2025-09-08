import type { JSONSchema } from '@dmitryrechkin/json-schema-to-zod'
import type { StructuredToolInterface } from '@langchain/core/tools'
import type {
  CallToolResult,
  EmbeddedResource,
  ImageContent,
  Tool as MCPTool,
  TextContent,
} from '@modelcontextprotocol/sdk/types.js'
import type { ZodTypeAny } from 'zod'
import type { BaseConnector } from '../connectors/base.js'

import { JSONSchemaToZod } from '@dmitryrechkin/json-schema-to-zod'
import { DynamicStructuredTool } from '@langchain/core/tools'
import { z } from 'zod'
import { logger } from '../logging.js'
import { BaseAdapter } from './base.js'

/**
 * A custom tool class that extends DynamicStructuredTool to automatically
 * unwrap the double-wrapped `input` parameter that can be erroneously
 * produced by LLMs when a tool's output (a JSON string) is used as the
 * input for a subsequent tool call.
 */
class MCPStructuredTool extends DynamicStructuredTool {
  async call(
    arg: Record<string, any> | string,
    ...rest: any[]
  ): Promise<string> {
    let unwrappedArg = arg
    // Check for the specific double-wrapping pattern: an object with a single
    // key 'input' which is a stringified JSON.
    if (
      typeof arg === 'object'
      && arg !== null
      && Object.keys(arg).length === 1
      && 'input' in arg
      && typeof arg.input === 'string'
    ) {
      try {
        // Attempt to parse the inner string. If successful, use the
        // parsed object as the actual argument.
        unwrappedArg = JSON.parse(arg.input)
        logger.debug('Unwrapped tool input from:', arg)
        logger.debug('Unwrapped tool input to:', unwrappedArg)
      }
      catch (e) {
        // Not a valid JSON string, proceed with the original argument.
        logger.debug('Input appeared double-wrapped, but failed to parse inner string.', e)
      }
    }
    return super.call(unwrappedArg, ...rest)
  }
}

function schemaToZod(schema: unknown): ZodTypeAny {
  try {
    return JSONSchemaToZod.convert(schema as JSONSchema)
  }
  catch (err) {
    logger.warn(`Failed to convert JSON schema to Zod: ${err}`)
    return z.any()
  }
}

function parseMcpToolResult(toolResult: CallToolResult): string {
  if (toolResult.isError) {
    throw new Error(`Tool execution failed: ${toolResult.content}`)
  }
  if (!toolResult.content || toolResult.content.length === 0) {
    throw new Error('Tool execution returned no content')
  }

  let decoded = ''
  for (const item of toolResult.content) {
    switch (item.type) {
      case 'text': {
        decoded += (item as TextContent).text
        break
      }
      case 'image': {
        decoded += (item as ImageContent).data
        break
      }
      case 'resource': {
        const res = (item as EmbeddedResource).resource
        if (res?.text !== undefined) {
          decoded += res.text
        }
        else if (res?.blob !== undefined) {
          // eslint-disable-next-line node/prefer-global/buffer
          decoded += res.blob instanceof Uint8Array || res.blob instanceof Buffer
            // eslint-disable-next-line node/prefer-global/buffer
            ? Buffer.from(res.blob).toString('base64')
            : String(res.blob)
        }
        else {
          throw new Error(`Unexpected resource type: ${res?.type}`)
        }
        break
      }
      default:
        throw new Error(`Unexpected content type: ${(item as any).type}`)
    }
  }
  return decoded
}

export class LangChainAdapter extends BaseAdapter<StructuredToolInterface> {
  constructor(disallowedTools: string[] = []) {
    super(disallowedTools)
  }

  /**
   * Convert a single MCP tool specification into a LangChainJS structured tool.
   */
  protected convertTool(
    mcpTool: MCPTool,
    connector: BaseConnector,
  ): StructuredToolInterface | null {
    // Filter out disallowed tools early.
    if (this.disallowedTools.includes(mcpTool.name)) {
      return null
    }

    // Derive a strict Zod schema for the tool's arguments.
    const argsSchema: ZodTypeAny = mcpTool.inputSchema
      ? schemaToZod(mcpTool.inputSchema)
      : z.object({}).optional()

    const tool = new MCPStructuredTool({
      name: mcpTool.name ?? 'NO NAME',
      description: mcpTool.description ?? '', // Blank is acceptable but discouraged.
      schema: argsSchema,
      func: async (input: Record<string, any>): Promise<string> => {
        logger.debug(`MCP tool \"${mcpTool.name}\" received input: ${JSON.stringify(input)}`)
        try {
          const result: CallToolResult = await connector.callTool(mcpTool.name, input)
          return parseMcpToolResult(result)
        }
        catch (err: any) {
          logger.error(`Error executing MCP tool: ${err.message}`)
          return `Error executing MCP tool: ${String(err)}`
        }
      },
    })

    return tool
  }
}
