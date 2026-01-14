package admin

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/jmoiron/sqlx"
	"github.com/khryptorgraphics/novacron/backend/core/security"
	"github.com/khryptorgraphics/novacron/backend/pkg/logger"
)

type DatabaseHandlers struct {
	db        *sql.DB
	protector *security.SQLInjectionProtector
	validator *security.InputValidator
}

type TableInfo struct {
	Name        string `json:"name"`
	Schema      string `json:"schema"`
	RowCount    int64  `json:"row_count"`
	Size        string `json:"size"`
	Description string `json:"description"`
}

type ColumnInfo struct {
	Name         string      `json:"name"`
	Type         string      `json:"type"`
	Nullable     bool        `json:"nullable"`
	DefaultValue interface{} `json:"default_value"`
	IsPrimaryKey bool        `json:"is_primary_key"`
}

type QueryResult struct {
	Columns  []string        `json:"columns"`
	Rows     [][]interface{} `json:"rows"`
	Affected int64           `json:"affected"`
	Duration string          `json:"duration"`
}

type QueryRequest struct {
	SQL string `json:"sql"`
}

type TableDetailsResponse struct {
	Table   TableInfo    `json:"table"`
	Columns []ColumnInfo `json:"columns"`
	Indexes []IndexInfo  `json:"indexes"`
}

type IndexInfo struct {
	Name    string   `json:"name"`
	Columns []string `json:"columns"`
	Unique  bool     `json:"unique"`
}

func NewDatabaseHandlers(db *sql.DB) *DatabaseHandlers {
	sqlxDB := sqlx.NewDb(db, "postgres")
	return &DatabaseHandlers{
		db:        db,
		protector: security.NewSQLInjectionProtector(sqlxDB),
		validator: security.NewInputValidator(),
	}
}

// GET /api/admin/database/tables - List all tables
func (h *DatabaseHandlers) ListTables(w http.ResponseWriter, r *http.Request) {
	query := `
		SELECT 
			t.table_name,
			t.table_schema,
			COALESCE(s.n_tup_ins + s.n_tup_upd + s.n_tup_del, 0) as row_count,
			pg_size_pretty(pg_total_relation_size(c.oid)) as size,
			obj_description(c.oid) as description
		FROM information_schema.tables t
		LEFT JOIN pg_class c ON c.relname = t.table_name
		LEFT JOIN pg_stat_user_tables s ON s.relname = t.table_name
		WHERE t.table_schema NOT IN ('information_schema', 'pg_catalog')
		AND t.table_type = 'BASE TABLE'
		ORDER BY t.table_name
	`

	rows, err := h.db.Query(query)
	if err != nil {
		logger.Error("Failed to query tables", "error", err)
		http.Error(w, "Internal server error", http.StatusInternalServerError)
		return
	}
	defer rows.Close()

	tables := []TableInfo{}
	for rows.Next() {
		var table TableInfo
		var description sql.NullString

		err := rows.Scan(&table.Name, &table.Schema, &table.RowCount, &table.Size, &description)
		if err != nil {
			logger.Error("Failed to scan table info", "error", err)
			continue
		}

		if description.Valid {
			table.Description = description.String
		}

		tables = append(tables, table)
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"tables": tables,
		"count":  len(tables),
	})
}

// GET /api/admin/database/tables/{table} - Get table details
func (h *DatabaseHandlers) GetTableDetails(w http.ResponseWriter, r *http.Request) {
	tableName := r.URL.Path[len("/api/admin/database/tables/"):]
	if tableName == "" {
		http.Error(w, "Table name is required", http.StatusBadRequest)
		return
	}

	// Sanitize table name to prevent SQL injection
	if !isValidIdentifier(tableName) {
		http.Error(w, "Invalid table name", http.StatusBadRequest)
		return
	}

	// Get table info
	var table TableInfo
	var description sql.NullString
	err := h.db.QueryRow(`
		SELECT 
			t.table_name,
			t.table_schema,
			COALESCE(s.n_tup_ins + s.n_tup_upd + s.n_tup_del, 0) as row_count,
			pg_size_pretty(pg_total_relation_size(c.oid)) as size,
			obj_description(c.oid) as description
		FROM information_schema.tables t
		LEFT JOIN pg_class c ON c.relname = t.table_name
		LEFT JOIN pg_stat_user_tables s ON s.relname = t.table_name
		WHERE t.table_name = $1 
		AND t.table_schema NOT IN ('information_schema', 'pg_catalog')
	`, tableName).Scan(&table.Name, &table.Schema, &table.RowCount, &table.Size, &description)

	if err != nil {
		if err == sql.ErrNoRows {
			http.Error(w, "Table not found", http.StatusNotFound)
		} else {
			logger.Error("Failed to get table info", "error", err)
			http.Error(w, "Internal server error", http.StatusInternalServerError)
		}
		return
	}

	if description.Valid {
		table.Description = description.String
	}

	// Get columns
	columnRows, err := h.db.Query(`
		SELECT 
			c.column_name,
			c.data_type,
			c.is_nullable = 'YES' as nullable,
			c.column_default,
			CASE WHEN pk.column_name IS NOT NULL THEN true ELSE false END as is_primary_key
		FROM information_schema.columns c
		LEFT JOIN (
			SELECT ku.column_name
			FROM information_schema.table_constraints tc
			JOIN information_schema.key_column_usage ku ON tc.constraint_name = ku.constraint_name
			WHERE tc.table_name = $1 AND tc.constraint_type = 'PRIMARY KEY'
		) pk ON pk.column_name = c.column_name
		WHERE c.table_name = $1
		ORDER BY c.ordinal_position
	`, tableName)

	if err != nil {
		logger.Error("Failed to get table columns", "error", err)
		http.Error(w, "Internal server error", http.StatusInternalServerError)
		return
	}
	defer columnRows.Close()

	columns := []ColumnInfo{}
	for columnRows.Next() {
		var column ColumnInfo
		var defaultValue sql.NullString

		err := columnRows.Scan(&column.Name, &column.Type, &column.Nullable, &defaultValue, &column.IsPrimaryKey)
		if err != nil {
			logger.Error("Failed to scan column info", "error", err)
			continue
		}

		if defaultValue.Valid {
			column.DefaultValue = defaultValue.String
		}

		columns = append(columns, column)
	}

	// Get indexes
	indexRows, err := h.db.Query(`
		SELECT 
			i.relname as index_name,
			array_agg(a.attname ORDER BY a.attnum) as columns,
			ix.indisunique as is_unique
		FROM pg_class i
		JOIN pg_index ix ON i.oid = ix.indexrelid
		JOIN pg_class t ON t.oid = ix.indrelid
		JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = ANY(ix.indkey)
		WHERE t.relname = $1
		GROUP BY i.relname, ix.indisunique
		ORDER BY i.relname
	`, tableName)

	if err != nil {
		logger.Error("Failed to get table indexes", "error", err)
		// Continue without indexes if query fails
	}

	indexes := []IndexInfo{}
	if indexRows != nil {
		defer indexRows.Close()
		for indexRows.Next() {
			var index IndexInfo
			var columnsArray []string

			err := indexRows.Scan(&index.Name, &columnsArray, &index.Unique)
			if err != nil {
				logger.Error("Failed to scan index info", "error", err)
				continue
			}

			index.Columns = columnsArray
			indexes = append(indexes, index)
		}
	}

	response := TableDetailsResponse{
		Table:   table,
		Columns: columns,
		Indexes: indexes,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// POST /api/admin/database/query - Execute read query
func (h *DatabaseHandlers) ExecuteQuery(w http.ResponseWriter, r *http.Request) {
	var req QueryRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	if req.SQL == "" {
		http.Error(w, "SQL query is required", http.StatusBadRequest)
		return
	}

	// Basic SQL injection prevention - only allow SELECT statements
	trimmedSQL := strings.TrimSpace(strings.ToUpper(req.SQL))
	if !strings.HasPrefix(trimmedSQL, "SELECT") && !strings.HasPrefix(trimmedSQL, "WITH") {
		http.Error(w, "Only SELECT queries are allowed", http.StatusBadRequest)
		return
	}

	// Check for dangerous keywords
	dangerousKeywords := []string{"DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE", "TRUNCATE"}
	for _, keyword := range dangerousKeywords {
		if strings.Contains(trimmedSQL, keyword) {
			http.Error(w, fmt.Sprintf("Keyword '%s' is not allowed in read-only queries", keyword), http.StatusBadRequest)
			return
		}
	}

	// Execute query with timeout
	ctx := r.Context()
	rows, err := h.db.QueryContext(ctx, req.SQL)
	if err != nil {
		logger.Error("Failed to execute query", "error", err, "sql", req.SQL)
		http.Error(w, fmt.Sprintf("Query execution failed: %v", err), http.StatusBadRequest)
		return
	}
	defer rows.Close()

	// Get column names
	columns, err := rows.Columns()
	if err != nil {
		logger.Error("Failed to get column names", "error", err)
		http.Error(w, "Internal server error", http.StatusInternalServerError)
		return
	}

	// Read all rows
	var resultRows [][]interface{}
	for rows.Next() {
		// Create slice of interface{} for each row
		values := make([]interface{}, len(columns))
		valuePtrs := make([]interface{}, len(columns))
		for i := range values {
			valuePtrs[i] = &values[i]
		}

		if err := rows.Scan(valuePtrs...); err != nil {
			logger.Error("Failed to scan row", "error", err)
			continue
		}

		// Convert []byte to string for JSON serialization
		for i, val := range values {
			if b, ok := val.([]byte); ok {
				values[i] = string(b)
			}
		}

		resultRows = append(resultRows, values)
	}

	result := QueryResult{
		Columns:  columns,
		Rows:     resultRows,
		Affected: int64(len(resultRows)),
		Duration: "N/A", // Could add timing if needed
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(result)
}

// POST /api/admin/database/execute - Execute write query (admin only)
func (h *DatabaseHandlers) ExecuteStatement(w http.ResponseWriter, r *http.Request) {
	var req QueryRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	if req.SQL == "" {
		http.Error(w, "SQL statement is required", http.StatusBadRequest)
		return
	}

	// Execute statement
	ctx := r.Context()
	result, err := h.db.ExecContext(ctx, req.SQL)
	if err != nil {
		logger.Error("Failed to execute statement", "error", err, "sql", req.SQL)
		http.Error(w, fmt.Sprintf("Statement execution failed: %v", err), http.StatusBadRequest)
		return
	}

	affected, _ := result.RowsAffected()

	response := QueryResult{
		Columns:  []string{},
		Rows:     [][]interface{}{},
		Affected: affected,
		Duration: "N/A",
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// Utility function to validate SQL identifiers
func isValidIdentifier(name string) bool {
	if len(name) == 0 || len(name) > 63 {
		return false
	}

	// Must start with letter or underscore
	if !((name[0] >= 'a' && name[0] <= 'z') ||
		(name[0] >= 'A' && name[0] <= 'Z') ||
		name[0] == '_') {
		return false
	}

	// Rest must be alphanumeric or underscore
	for i := 1; i < len(name); i++ {
		c := name[i]
		if !((c >= 'a' && c <= 'z') ||
			(c >= 'A' && c <= 'Z') ||
			(c >= '0' && c <= '9') ||
			c == '_') {
			return false
		}
	}

	return true
}
