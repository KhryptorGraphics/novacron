package main

import (
	"database/sql"
	"flag"
	"fmt"
	"log"
	"os"

	_ "github.com/lib/pq"

	"github.com/khryptorgraphics/novacron/backend/migrations"
)

func main() {
	action := flag.String("action", "up", "migration action: up, down, version")
	databaseURL := flag.String("database-url", os.Getenv("DATABASE_URL"), "PostgreSQL connection URL")
	databaseName := flag.String("database-name", "novacron", "database name for migration bookkeeping")
	flag.Parse()

	if *databaseURL == "" {
		log.Fatal("database URL is required via -database-url or DATABASE_URL")
	}

	db, err := sql.Open("postgres", *databaseURL)
	if err != nil {
		log.Fatalf("open database: %v", err)
	}
	defer db.Close()

	if err := db.Ping(); err != nil {
		log.Fatalf("ping database: %v", err)
	}

	manager, err := migrations.NewManager(db, *databaseName)
	if err != nil {
		log.Fatalf("create migration manager: %v", err)
	}

	switch *action {
	case "up":
		err = manager.Up()
	case "down":
		err = manager.Down()
	case "version":
		var version uint
		var dirty bool
		version, dirty, err = manager.Version()
		if err == nil {
			fmt.Printf("version=%d dirty=%t\n", version, dirty)
		}
	default:
		log.Fatalf("unsupported action %q", *action)
	}
	if err != nil {
		log.Fatalf("migration %s failed: %v", *action, err)
	}
}
