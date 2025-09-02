package output

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"reflect"
	"strings"
	"text/tabwriter"

	"github.com/fatih/color"
	"github.com/olekukonko/tablewriter"
	"github.com/spf13/viper"
	"gopkg.in/yaml.v3"
)

// Format represents output format
type Format string

const (
	FormatTable Format = "table"
	FormatJSON  Format = "json"
	FormatYAML  Format = "yaml"
	FormatWide  Format = "wide"
	FormatCustom Format = "custom"
)

// GetFormat returns the current output format
func GetFormat() Format {
	format := viper.GetString("output")
	switch format {
	case "json":
		return FormatJSON
	case "yaml":
		return FormatYAML
	case "wide":
		return FormatWide
	case "custom":
		return FormatCustom
	default:
		return FormatTable
	}
}

// Printer handles output formatting
type Printer struct {
	format Format
	writer io.Writer
	noColor bool
}

// NewPrinter creates a new printer
func NewPrinter(format Format) *Printer {
	return &Printer{
		format:  format,
		writer:  os.Stdout,
		noColor: viper.GetBool("no-color"),
	}
}

// WithWriter sets a custom writer
func (p *Printer) WithWriter(w io.Writer) *Printer {
	p.writer = w
	return p
}

// Print prints the data in the configured format
func (p *Printer) Print(data interface{}) error {
	switch p.format {
	case FormatJSON:
		return p.printJSON(data)
	case FormatYAML:
		return p.printYAML(data)
	case FormatTable, FormatWide:
		return p.printTable(data)
	default:
		return fmt.Errorf("unsupported format: %s", p.format)
	}
}

// printJSON prints data as JSON
func (p *Printer) printJSON(data interface{}) error {
	encoder := json.NewEncoder(p.writer)
	encoder.SetIndent("", "  ")
	return encoder.Encode(data)
}

// printYAML prints data as YAML
func (p *Printer) printYAML(data interface{}) error {
	encoder := yaml.NewEncoder(p.writer)
	encoder.SetIndent(2)
	return encoder.Encode(data)
}

// printTable prints data as a table
func (p *Printer) printTable(data interface{}) error {
	// Handle different data types
	v := reflect.ValueOf(data)
	
	switch v.Kind() {
	case reflect.Slice, reflect.Array:
		return p.printSliceTable(v)
	case reflect.Struct:
		return p.printStructTable(v)
	case reflect.Map:
		return p.printMapTable(v)
	default:
		// Fall back to simple print
		fmt.Fprintln(p.writer, data)
		return nil
	}
}

// printSliceTable prints a slice as a table
func (p *Printer) printSliceTable(v reflect.Value) error {
	if v.Len() == 0 {
		fmt.Fprintln(p.writer, "No items found")
		return nil
	}

	// Get headers from first element
	first := v.Index(0)
	if first.Kind() != reflect.Struct {
		// Simple slice, print as list
		for i := 0; i < v.Len(); i++ {
			fmt.Fprintln(p.writer, v.Index(i).Interface())
		}
		return nil
	}

	// Create table
	table := tablewriter.NewWriter(p.writer)
	
	// Configure table style
	table.SetBorder(false)
	table.SetHeaderLine(false)
	table.SetColumnSeparator("")
	table.SetHeaderAlignment(tablewriter.ALIGN_LEFT)
	table.SetAlignment(tablewriter.ALIGN_LEFT)

	// Get headers
	headers := p.getStructHeaders(first.Type())
	table.SetHeader(headers)

	// Add rows
	for i := 0; i < v.Len(); i++ {
		row := p.getStructValues(v.Index(i))
		table.Append(row)
	}

	table.Render()
	return nil
}

// printStructTable prints a struct as a table
func (p *Printer) printStructTable(v reflect.Value) error {
	table := tablewriter.NewWriter(p.writer)
	
	// Configure table style
	table.SetBorder(false)
	table.SetHeaderLine(false)
	table.SetColumnSeparator("")
	table.SetAlignment(tablewriter.ALIGN_LEFT)

	// Set headers
	table.SetHeader([]string{"FIELD", "VALUE"})

	// Add fields
	t := v.Type()
	for i := 0; i < v.NumField(); i++ {
		field := t.Field(i)
		value := v.Field(i)
		
		// Skip unexported fields
		if !field.IsExported() {
			continue
		}

		// Get field name from json tag if available
		name := field.Name
		if tag := field.Tag.Get("json"); tag != "" {
			parts := strings.Split(tag, ",")
			if parts[0] != "-" {
				name = parts[0]
			}
		}

		// Format value
		valueStr := p.formatValue(value.Interface())
		
		table.Append([]string{name, valueStr})
	}

	table.Render()
	return nil
}

// printMapTable prints a map as a table
func (p *Printer) printMapTable(v reflect.Value) error {
	table := tablewriter.NewWriter(p.writer)
	
	// Configure table style
	table.SetBorder(false)
	table.SetHeaderLine(false)
	table.SetColumnSeparator("")
	table.SetAlignment(tablewriter.ALIGN_LEFT)

	// Set headers
	table.SetHeader([]string{"KEY", "VALUE"})

	// Add entries
	for _, key := range v.MapKeys() {
		value := v.MapIndex(key)
		keyStr := p.formatValue(key.Interface())
		valueStr := p.formatValue(value.Interface())
		table.Append([]string{keyStr, valueStr})
	}

	table.Render()
	return nil
}

// getStructHeaders gets headers from a struct type
func (p *Printer) getStructHeaders(t reflect.Type) []string {
	var headers []string
	
	for i := 0; i < t.NumField(); i++ {
		field := t.Field(i)
		
		// Skip unexported fields
		if !field.IsExported() {
			continue
		}

		// Skip fields with json:"-"
		if tag := field.Tag.Get("json"); tag == "-" {
			continue
		}

		// Get field name from json tag if available
		name := field.Name
		if tag := field.Tag.Get("json"); tag != "" {
			parts := strings.Split(tag, ",")
			if parts[0] != "-" {
				name = strings.ToUpper(parts[0])
			}
		}

		// Skip complex types in table view unless wide format
		if p.format != FormatWide {
			switch field.Type.Kind() {
			case reflect.Struct, reflect.Slice, reflect.Map:
				if field.Type.String() != "time.Time" {
					continue
				}
			}
		}

		headers = append(headers, name)
	}

	return headers
}

// getStructValues gets values from a struct
func (p *Printer) getStructValues(v reflect.Value) []string {
	var values []string
	t := v.Type()

	for i := 0; i < v.NumField(); i++ {
		field := t.Field(i)
		value := v.Field(i)

		// Skip unexported fields
		if !field.IsExported() {
			continue
		}

		// Skip fields with json:"-"
		if tag := field.Tag.Get("json"); tag == "-" {
			continue
		}

		// Skip complex types in table view unless wide format
		if p.format != FormatWide {
			switch field.Type.Kind() {
			case reflect.Struct, reflect.Slice, reflect.Map:
				if field.Type.String() != "time.Time" {
					continue
				}
			}
		}

		values = append(values, p.formatValue(value.Interface()))
	}

	return values
}

// formatValue formats a value for display
func (p *Printer) formatValue(v interface{}) string {
	if v == nil {
		return "<none>"
	}

	switch val := v.(type) {
	case string:
		if val == "" {
			return "<none>"
		}
		return val
	case bool:
		if val {
			if !p.noColor {
				return color.GreenString("true")
			}
			return "true"
		}
		if !p.noColor {
			return color.RedString("false")
		}
		return "false"
	case []string:
		if len(val) == 0 {
			return "<none>"
		}
		return strings.Join(val, ", ")
	case map[string]string:
		if len(val) == 0 {
			return "<none>"
		}
		var pairs []string
		for k, v := range val {
			pairs = append(pairs, fmt.Sprintf("%s=%s", k, v))
		}
		return strings.Join(pairs, ", ")
	default:
		// Use default formatting
		s := fmt.Sprintf("%v", v)
		if s == "" {
			return "<none>"
		}
		return s
	}
}

// Success prints a success message
func Success(format string, args ...interface{}) {
	if viper.GetBool("no-color") {
		fmt.Printf("✓ "+format+"\n", args...)
	} else {
		color.Green("✓ " + fmt.Sprintf(format, args...))
	}
}

// Error prints an error message
func Error(format string, args ...interface{}) {
	if viper.GetBool("no-color") {
		fmt.Fprintf(os.Stderr, "✗ "+format+"\n", args...)
	} else {
		color.Red("✗ " + fmt.Sprintf(format, args...))
	}
}

// Warning prints a warning message
func Warning(format string, args ...interface{}) {
	if viper.GetBool("no-color") {
		fmt.Printf("⚠ "+format+"\n", args...)
	} else {
		color.Yellow("⚠ " + fmt.Sprintf(format, args...))
	}
}

// Info prints an info message
func Info(format string, args ...interface{}) {
	fmt.Printf("ℹ "+format+"\n", args...)
}

// PrintHeader prints a section header
func PrintHeader(text string) {
	if viper.GetBool("no-color") {
		fmt.Printf("\n=== %s ===\n\n", text)
	} else {
		color.Cyan("\n=== %s ===\n\n", text)
	}
}

// TabWriter creates a new tab writer for aligned output
func TabWriter() *tabwriter.Writer {
	return tabwriter.NewWriter(os.Stdout, 0, 8, 2, ' ', 0)
}