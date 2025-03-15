package main

import (
	"fmt"
	"os"

	"github.com/khryptorgraphics/novacron/backend/examples/backup/examples"
)

func main() {
	// Print available examples
	fmt.Println("Novacron Backup System Examples")
	fmt.Println("===============================")
	fmt.Println("Available examples:")
	fmt.Println("1. Local Provider Example")
	fmt.Println("2. Enhancement Combinations")
	fmt.Println()

	// Get user input or command line argument
	exampleNumber := 0
	if len(os.Args) > 1 {
		_, err := fmt.Sscanf(os.Args[1], "%d", &exampleNumber)
		if err != nil {
			fmt.Printf("Invalid example number: %s\n", os.Args[1])
			os.Exit(1)
		}
	} else {
		fmt.Print("Enter example number to run: ")
		_, err := fmt.Scanf("%d", &exampleNumber)
		if err != nil {
			fmt.Printf("Invalid input: %v\n", err)
			os.Exit(1)
		}
	}

	// Run the selected example
	switch exampleNumber {
	case 1:
		fmt.Println("\nRunning Local Provider Example...")
		examples.RunLocalProviderExample()
	case 2:
		fmt.Println("\nRunning Enhancement Combinations Examples...")
		examples.DemonstrateEnhancementCombinations()
	default:
		fmt.Printf("Invalid example number: %d\n", exampleNumber)
		os.Exit(1)
	}
}
