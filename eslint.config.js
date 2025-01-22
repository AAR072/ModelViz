import js from '@eslint/js';
import { includeIgnoreFile } from '@eslint/compat';
import svelte from 'eslint-plugin-svelte';
import globals from 'globals';
import { fileURLToPath } from 'node:url';
import ts from 'typescript-eslint';
const gitignorePath = fileURLToPath(new URL("./.gitignore", import.meta.url));

export default ts.config(
  includeIgnoreFile(gitignorePath),
  js.configs.recommended,
  ts.configs.recommended,
  ...svelte.configs["flat/recommended"],
  {
    languageOptions: {
      globals: {
        ...globals.browser,
        ...globals.node
      }
    }
  },
  {
    files: ["**/*.svelte"],

    languageOptions: {
      parserOptions: {
        parser: ts.parser
      }
    }
  },
  {
    "rules": {
      // File Names
      "filenames/match-regex": "off",

      // Formatting and Braces
      "curly": ["error", "multi-line"],
      "semi": ["error", "always"],

      // Function Arguments (4-space indentation on line wrap, no rule available here, so manual review is needed)

      // Language Features - Identifiers
      "camelcase": ["error", { "properties": "never" }],
      "@typescript-eslint/naming-convention": [
        "error",
        { "selector": "variable", "format": ["camelCase", "UPPER_CASE"] },
        { "selector": "function", "format": ["camelCase"] },
        { "selector": "class", "format": ["PascalCase"] },
        { "selector": "interface", "format": ["PascalCase"] },
        { "selector": "enum", "format": ["PascalCase"] },
        { "selector": "enumMember", "format": ["UPPER_CASE"] },
        { "selector": "typeParameter", "format": ["PascalCase"] }
      ],

      // Constants
      "prefer-const": "error",
      "one-var": ["error", "never"],

      // Type Declarations
      "@typescript-eslint/explicit-function-return-type": [
        "error",
        { "allowExpressions": true, 
          "allowTypedFunctionExpressions": true } 
      ],
      "@typescript-eslint/typedef": [
        "error",
        {
          arrayDestructuring: true,
          arrowParameter: false,
          memberVariableDeclaration: true,
          objectDestructuring: true,
          parameter: true,
          propertyDeclaration: true,
          variableDeclaration: true,
          variableDeclarationIgnoreFunction: false
        }
      ],

      // Private Fields
      "no-restricted-syntax": [
        "error",
        {
          "selector": "PrivateIdentifier",
          "message": "Use TypeScript's private modifier instead of # syntax for private fields."
        }
      ],
      "@typescript-eslint/no-inferrable-types": "off",
      "no-unused-vars": "off",
      "@typescript-eslint/no-unused-vars": "off",

      // Trailing Commas
      "comma-dangle": [
        "error",
        {
          "arrays": "never",
          "objects": "never",
          "imports": "never",
          "exports": "never",
          "functions": "never"
        }
      ]
    }
  }
);
