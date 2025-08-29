#!/usr/bin/env node

/**
 * NovaCron Command Registry
 * Central registry for all Claude slash commands
 */

const fs = require('fs');
const path = require('path');

class CommandRegistry {
  constructor() {
    this.commands = new Map();
    this.categories = new Set();
    this.loadCommands();
  }

  loadCommands() {
    const commandsDir = __dirname;
    const categories = fs.readdirSync(commandsDir, { withFileTypes: true })
      .filter(dirent => dirent.isDirectory())
      .map(dirent => dirent.name);

    categories.forEach(category => {
      this.categories.add(category);
      const categoryPath = path.join(commandsDir, category);
      
      fs.readdirSync(categoryPath)
        .filter(file => file.endsWith('.js'))
        .forEach(file => {
          try {
            const commandPath = path.join(categoryPath, file);
            const command = require(commandPath);
            
            if (command.name) {
              this.commands.set(command.name, {
                ...command,
                category,
                path: commandPath
              });
            }
          } catch (error) {
            console.error(`Failed to load command ${file}:`, error.message);
          }
        });
    });

    console.log(`✅ Loaded ${this.commands.size} commands across ${this.categories.size} categories`);
  }

  getCommand(name) {
    // Try exact match first
    if (this.commands.has(name)) {
      return this.commands.get(name);
    }

    // Try with slash prefix
    if (!name.startsWith('/') && this.commands.has('/' + name)) {
      return this.commands.get('/' + name);
    }

    // Try without slash prefix
    if (name.startsWith('/') && this.commands.has(name.substring(1))) {
      return this.commands.get(name.substring(1));
    }

    return null;
  }

  listCommands(category = null) {
    const commands = Array.from(this.commands.values());
    
    if (category) {
      return commands.filter(cmd => cmd.category === category);
    }
    
    return commands;
  }

  listCategories() {
    return Array.from(this.categories);
  }

  async executeCommand(commandName, args = []) {
    const command = this.getCommand(commandName);
    
    if (!command) {
      return {
        error: `Command '${commandName}' not found`,
        suggestion: `Available commands: ${Array.from(this.commands.keys()).join(', ')}`
      };
    }

    try {
      console.log(`\n🚀 Executing: ${command.name}`);
      
      if (command.execute) {
        return await command.execute(args);
      } else {
        return {
          error: `Command '${commandName}' does not have an execute method`
        };
      }
    } catch (error) {
      return {
        error: `Command execution failed: ${error.message}`,
        stack: error.stack
      };
    }
  }

  getHelp(commandName = null) {
    if (commandName) {
      const command = this.getCommand(commandName);
      if (command) {
        return {
          name: command.name,
          description: command.description,
          usage: command.usage,
          category: command.category
        };
      } else {
        return { error: `Command '${commandName}' not found` };
      }
    }

    // Return all commands grouped by category
    const helpText = {
      title: 'NovaCron Commands',
      categories: {}
    };

    this.categories.forEach(category => {
      helpText.categories[category] = this.listCommands(category).map(cmd => ({
        name: cmd.name,
        description: cmd.description
      }));
    });

    return helpText;
  }
}

// Export singleton instance
module.exports = new CommandRegistry();

// CLI interface if run directly
if (require.main === module) {
  const registry = module.exports;
  const args = process.argv.slice(2);
  
  if (args.length === 0 || args[0] === 'help') {
    console.log('\n📚 NovaCron Command System');
    const help = registry.getHelp();
    
    Object.entries(help.categories).forEach(([category, commands]) => {
      console.log(`\n📁 ${category.toUpperCase()}`);
      commands.forEach(cmd => {
        console.log(`  ${cmd.name.padEnd(25)} - ${cmd.description}`);
      });
    });
    
    console.log('\n💡 Usage: node index.js <command> [args...]');
    console.log('   Example: node index.js vm:migrate vm-123 node-2 --type live\n');
  } else {
    const commandName = args[0];
    const commandArgs = args.slice(1);
    
    registry.executeCommand(commandName, commandArgs).then(result => {
      if (result.error) {
        console.error('\n❌ Error:', result.error);
        if (result.suggestion) {
          console.log('💡 Suggestion:', result.suggestion);
        }
        process.exit(1);
      } else {
        console.log('\n✅ Command completed successfully');
        if (result.success !== undefined) {
          console.log('Result:', JSON.stringify(result, null, 2));
        }
      }
    }).catch(error => {
      console.error('\n❌ Unexpected error:', error);
      process.exit(1);
    });
  }
}