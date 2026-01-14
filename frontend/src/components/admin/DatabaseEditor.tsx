"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { 
  Table, 
  TableBody, 
  TableCell, 
  TableHead, 
  TableHeader, 
  TableRow 
} from "@/components/ui/table";
import { 
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Switch } from "@/components/ui/switch";
import { FadeIn } from "@/lib/animations";
import { 
  Database, 
  Search, 
  Edit, 
  Trash2, 
  Plus, 
  Save, 
  X, 
  AlertTriangle,
  Eye,
  Download,
  Upload,
  RefreshCw
} from "lucide-react";
import { cn } from "@/lib/utils";

// Mock database schema
const databaseTables = [
  { name: "users", rows: 1247, size: "24.3 MB", description: "User accounts and profiles" },
  { name: "organizations", rows: 89, size: "2.1 MB", description: "Organization entities" },
  { name: "vms", rows: 2341, size: "45.7 MB", description: "Virtual machine configurations" },
  { name: "schedules", rows: 5432, size: "12.8 MB", description: "Scheduled job definitions" },
  { name: "migrations", rows: 876, size: "156.2 MB", description: "VM migration history" },
  { name: "audit_logs", rows: 98765, size: "234.5 MB", description: "System audit trail" },
  { name: "sessions", rows: 456, size: "3.4 MB", description: "User session data" },
  { name: "configurations", rows: 123, size: "890 KB", description: "System configuration settings" }
];

// Mock user data for editing
const mockUsers = [
  { 
    id: 1, 
    email: "user@organization.com", 
    name: "John Doe", 
    role: "user", 
    status: "active", 
    created_at: "2024-01-15",
    last_login: "2024-08-20"
  },
  { 
    id: 2, 
    email: "admin@novacron.io", 
    name: "System Admin", 
    role: "admin", 
    status: "active", 
    created_at: "2024-01-01",
    last_login: "2024-08-24"
  },
  { 
    id: 3, 
    email: "jane@company.com", 
    name: "Jane Smith", 
    role: "moderator", 
    status: "suspended", 
    created_at: "2024-02-10",
    last_login: "2024-08-18"
  }
];

interface EditingRow {
  id: number | null;
  data: Record<string, any>;
}

export function DatabaseEditor() {
  const [selectedTable, setSelectedTable] = useState("users");
  const [tableData, setTableData] = useState(mockUsers);
  const [searchQuery, setSearchQuery] = useState("");
  const [editingRow, setEditingRow] = useState<EditingRow | null>(null);
  const [showDangerZone, setShowDangerZone] = useState(false);
  const [loading, setLoading] = useState(false);

  const filteredData = tableData.filter(row => 
    Object.values(row).some(value => 
      String(value).toLowerCase().includes(searchQuery.toLowerCase())
    )
  );

  const handleEdit = (row: any) => {
    setEditingRow({ id: row.id, data: { ...row } });
  };

  const handleSave = async () => {
    if (!editingRow) return;
    
    setLoading(true);
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    if (editingRow.id) {
      // Update existing row
      setTableData(prev => 
        prev.map(row => row.id === editingRow.id ? { ...editingRow.data } : row)
      );
    } else {
      // Add new row
      const newId = Math.max(...tableData.map(r => r.id)) + 1;
      setTableData(prev => [...prev, { ...editingRow.data, id: newId }]);
    }
    
    setEditingRow(null);
    setLoading(false);
  };

  const handleDelete = async (id: number) => {
    if (!confirm("Are you sure you want to delete this record? This action cannot be undone.")) {
      return;
    }
    
    setLoading(true);
    await new Promise(resolve => setTimeout(resolve, 1000));
    setTableData(prev => prev.filter(row => row.id !== id));
    setLoading(false);
  };

  const handleAddNew = () => {
    setEditingRow({
      id: null,
      data: {
        email: "",
        name: "",
        role: "user",
        status: "active",
        created_at: new Date().toISOString().split('T')[0],
        last_login: ""
      }
    });
  };

  const handleCancel = () => {
    setEditingRow(null);
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case "active": return "bg-green-500";
      case "suspended": return "bg-red-500";
      case "pending": return "bg-yellow-500";
      default: return "bg-gray-500";
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold flex items-center gap-2">
            <Database className="h-6 w-6" />
            Database Editor
          </h2>
          <p className="text-gray-600 dark:text-gray-400 mt-1">
            Direct database table management with CRUD operations
          </p>
        </div>
        
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm">
            <Download className="h-4 w-4 mr-2" />
            Export
          </Button>
          <Button variant="outline" size="sm">
            <Upload className="h-4 w-4 mr-2" />
            Import
          </Button>
        </div>
      </div>

      {/* Warning Banner */}
      <Card className="border-orange-200 bg-orange-50 dark:border-orange-800 dark:bg-orange-950">
        <CardContent className="p-4">
          <div className="flex items-center gap-2 text-orange-800 dark:text-orange-200">
            <AlertTriangle className="h-5 w-5" />
            <span className="font-medium">
              Warning: Direct database editing can affect system integrity. Use with caution.
            </span>
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Table Selection Sidebar */}
        <div className="lg:col-span-1">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Database Tables</CardTitle>
              <CardDescription>Select a table to view and edit</CardDescription>
            </CardHeader>
            <CardContent className="p-0">
              <div className="space-y-1">
                {databaseTables.map((table) => (
                  <button
                    key={table.name}
                    onClick={() => setSelectedTable(table.name)}
                    className={cn(
                      "w-full text-left px-4 py-3 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors",
                      selectedTable === table.name && "bg-blue-50 dark:bg-blue-950 border-r-2 border-blue-500"
                    )}
                  >
                    <div className="font-medium">{table.name}</div>
                    <div className="text-sm text-gray-600 dark:text-gray-400">
                      {table.rows.toLocaleString()} rows • {table.size}
                    </div>
                  </button>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Main Table Editor */}
        <div className="lg:col-span-3 space-y-4">
          {/* Controls */}
          <Card>
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <CardTitle className="capitalize">{selectedTable} Table</CardTitle>
                <div className="flex items-center gap-2">
                  <Button 
                    onClick={handleAddNew}
                    disabled={!!editingRow}
                    size="sm"
                  >
                    <Plus className="h-4 w-4 mr-2" />
                    Add New
                  </Button>
                  <Button variant="outline" size="sm">
                    <RefreshCw className="h-4 w-4 mr-2" />
                    Refresh
                  </Button>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="flex items-center gap-4 mb-4">
                <div className="relative flex-1">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
                  <Input
                    placeholder="Search records..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="pl-10"
                  />
                </div>
                <Badge variant="outline">
                  {filteredData.length} records
                </Badge>
              </div>

              {/* Data Table */}
              <div className="border rounded-md">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>ID</TableHead>
                      <TableHead>Email</TableHead>
                      <TableHead>Name</TableHead>
                      <TableHead>Role</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead>Created</TableHead>
                      <TableHead>Last Login</TableHead>
                      <TableHead>Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {/* Editing Row */}
                    {editingRow && (
                      <TableRow className="bg-blue-50 dark:bg-blue-950">
                        <TableCell>{editingRow.id || "New"}</TableCell>
                        <TableCell>
                          <Input
                            value={editingRow.data.email}
                            onChange={(e) => setEditingRow(prev => prev && {
                              ...prev,
                              data: { ...prev.data, email: e.target.value }
                            })}
                            className="h-8"
                          />
                        </TableCell>
                        <TableCell>
                          <Input
                            value={editingRow.data.name}
                            onChange={(e) => setEditingRow(prev => prev && {
                              ...prev,
                              data: { ...prev.data, name: e.target.value }
                            })}
                            className="h-8"
                          />
                        </TableCell>
                        <TableCell>
                          <Select
                            value={editingRow.data.role}
                            onValueChange={(value) => setEditingRow(prev => prev && {
                              ...prev,
                              data: { ...prev.data, role: value }
                            })}
                          >
                            <SelectTrigger className="h-8">
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="user">User</SelectItem>
                              <SelectItem value="moderator">Moderator</SelectItem>
                              <SelectItem value="admin">Admin</SelectItem>
                            </SelectContent>
                          </Select>
                        </TableCell>
                        <TableCell>
                          <Select
                            value={editingRow.data.status}
                            onValueChange={(value) => setEditingRow(prev => prev && {
                              ...prev,
                              data: { ...prev.data, status: value }
                            })}
                          >
                            <SelectTrigger className="h-8">
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="active">Active</SelectItem>
                              <SelectItem value="suspended">Suspended</SelectItem>
                              <SelectItem value="pending">Pending</SelectItem>
                            </SelectContent>
                          </Select>
                        </TableCell>
                        <TableCell>
                          <Input
                            type="date"
                            value={editingRow.data.created_at}
                            onChange={(e) => setEditingRow(prev => prev && {
                              ...prev,
                              data: { ...prev.data, created_at: e.target.value }
                            })}
                            className="h-8"
                          />
                        </TableCell>
                        <TableCell>
                          <Input
                            type="date"
                            value={editingRow.data.last_login}
                            onChange={(e) => setEditingRow(prev => prev && {
                              ...prev,
                              data: { ...prev.data, last_login: e.target.value }
                            })}
                            className="h-8"
                          />
                        </TableCell>
                        <TableCell>
                          <div className="flex items-center gap-1">
                            <Button
                              size="sm"
                              onClick={handleSave}
                              disabled={loading}
                            >
                              <Save className="h-3 w-3" />
                            </Button>
                            <Button
                              size="sm"
                              variant="outline"
                              onClick={handleCancel}
                            >
                              <X className="h-3 w-3" />
                            </Button>
                          </div>
                        </TableCell>
                      </TableRow>
                    )}

                    {/* Data Rows */}
                    {filteredData.map((row) => (
                      <TableRow key={row.id}>
                        <TableCell className="font-medium">{row.id}</TableCell>
                        <TableCell>{row.email}</TableCell>
                        <TableCell>{row.name}</TableCell>
                        <TableCell>
                          <Badge variant="outline" className="capitalize">
                            {row.role}
                          </Badge>
                        </TableCell>
                        <TableCell>
                          <div className="flex items-center gap-2">
                            <div className={cn("h-2 w-2 rounded-full", getStatusColor(row.status))} />
                            <span className="capitalize">{row.status}</span>
                          </div>
                        </TableCell>
                        <TableCell>{row.created_at}</TableCell>
                        <TableCell>{row.last_login || "Never"}</TableCell>
                        <TableCell>
                          <div className="flex items-center gap-1">
                            <Button
                              size="sm"
                              variant="ghost"
                              onClick={() => handleEdit(row)}
                              disabled={!!editingRow}
                            >
                              <Edit className="h-3 w-3" />
                            </Button>
                            <Button
                              size="sm"
                              variant="ghost"
                              onClick={() => handleDelete(row.id)}
                              disabled={!!editingRow || loading}
                              className="text-red-600 hover:text-red-700"
                            >
                              <Trash2 className="h-3 w-3" />
                            </Button>
                          </div>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </CardContent>
          </Card>

          {/* Danger Zone */}
          <Card className="border-red-200">
            <CardHeader>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-red-600">
                  <AlertTriangle className="h-5 w-5" />
                  <CardTitle>Danger Zone</CardTitle>
                </div>
                <Switch
                  checked={showDangerZone}
                  onCheckedChange={setShowDangerZone}
                />
              </div>
              <CardDescription>
                Destructive actions that cannot be undone
              </CardDescription>
            </CardHeader>
            {showDangerZone && (
              <CardContent>
                <div className="flex items-center gap-4">
                  <Button variant="destructive" size="sm">
                    <Trash2 className="h-4 w-4 mr-2" />
                    Truncate Table
                  </Button>
                  <Button variant="destructive" size="sm">
                    <Trash2 className="h-4 w-4 mr-2" />
                    Drop Table
                  </Button>
                  <span className="text-sm text-gray-600 dark:text-gray-400">
                    These actions require additional confirmation
                  </span>
                </div>
              </CardContent>
            )}
          </Card>
        </div>
      </div>
    </div>
  );
}