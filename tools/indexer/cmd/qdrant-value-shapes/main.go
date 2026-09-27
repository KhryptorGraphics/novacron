package main

import (
	"fmt"

	"github.com/qdrant/go-client/qdrant"
)

func main() {
	// Create a string value
	stringValue := &qdrant.Value{}
	fmt.Printf("String value fields: %+v\n", stringValue)

	// Try different ways to set the value
	fmt.Println("Trying different ways to set a string value:")

	v1 := &qdrant.Value{Kind: &qdrant.Value_StringValue{StringValue: "test"}}
	fmt.Printf("v1: %+v\n", v1)

	// Try vectors
	vec := make(map[string]*qdrant.Vector)
	vec["default"] = &qdrant.Vector{Data: []float32{1.0, 2.0, 3.0}}

	fmt.Printf("Vector: %+v\n", vec)

	// Try creating a point
	point := &qdrant.PointStruct{
		Id: &qdrant.PointId{
			PointIdOptions: &qdrant.PointId_Num{
				Num: 42,
			},
		},
		Vectors: &qdrant.Vectors{
			VectorsOptions: &qdrant.Vectors_Vectors{
				Vectors: &qdrant.NamedVectors{Vectors: vec},
			},
		},
	}

	fmt.Printf("Point: %+v\n", point)
}
