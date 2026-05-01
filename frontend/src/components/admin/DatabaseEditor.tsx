"use client";

import { useState } from "react";
import type { Dispatch, SetStateAction } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { FadeIn } from "@/lib/animations";
import { useCreateUser, useDeleteUser, useUpdateUser, useUsers } from "@/lib/api/hooks/useAdmin";
import type { User } from "@/lib/api/types";
import {
  AlertTriangle,
  CheckCircle,
  Database,
  Edit,
  Plus,
  RefreshCw,
  Save,
  Search,
  Trash2,
  X,
} from "lucide-react";
import { cn } from "@/lib/utils";

type CanonicalUser = Partial<User> & {
  username?: string;
  active?: boolean;
};

type UserRow = {
  id: string;
  username: string;
  email: string;
  role: string;
  active: boolean;
  created_at: string;
  updated_at: string;
};

type EditingRow = {
  id: string | null;
  data: {
    username: string;
    email: string;
    role: string;
    active: boolean;
    password: string;
  };
};

const liveTables = [
  {
    name: "users",
    description: "Canonical admin user accounts",
    contract: "GET/POST/PUT/DELETE /api/admin/users",
    live: true,
  },
  {
    name: "audit_logs",
    description: "Audit log read model",
    contract: "Pending canonical contract",
    live: false,
  },
  {
    name: "sessions",
    description: "User session read model",
    contract: "Pending canonical contract",
    live: false,
  },
  {
    name: "configurations",
    description: "Runtime configuration records",
    contract: "Pending canonical contract",
    live: false,
  },
];

const roleOptions = ["admin", "operator", "viewer", "user", "super-admin"];

function normalizeUser(user: CanonicalUser): UserRow {
  return {
    id: String(user.id ?? ""),
    username: user.username || user.name || user.email || "",
    email: user.email || "",
    role: user.role || "user",
    active: user.active !== false && user.status !== "disabled" && user.status !== "suspended",
    created_at: user.created_at || "",
    updated_at: user.updated_at || "",
  };
}

function formatDate(value: string) {
  if (!value) return "never";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString();
}

export function DatabaseEditor() {
  const [selectedTable, setSelectedTable] = useState("users");
  const [searchQuery, setSearchQuery] = useState("");
  const [editingRow, setEditingRow] = useState<EditingRow | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const userFilters: { search?: string; pageSize: number } = { pageSize: 100 };
  if (searchQuery) {
    userFilters.search = searchQuery;
  }
  const usersQuery = useUsers(userFilters);
  const createUser = useCreateUser();
  const updateUser = useUpdateUser();
  const deleteUser = useDeleteUser();

  const selectedTableInfo = liveTables.find((table) => table.name === selectedTable) || liveTables[0];
  const rows = (usersQuery.data?.users || []).map((user) => normalizeUser(user as CanonicalUser));
  const isMutating = createUser.isPending || updateUser.isPending || deleteUser.isPending;
  const loading = usersQuery.isLoading || isMutating;

  const filteredRows = rows.filter((row) => {
    const query = searchQuery.toLowerCase();
    return [row.id, row.username, row.email, row.role, row.active ? "active" : "disabled"]
      .some((value) => value.toLowerCase().includes(query));
  });

  const handleEdit = (row: UserRow) => {
    setNotice(null);
    setEditingRow({
      id: row.id,
      data: {
        username: row.username,
        email: row.email,
        role: row.role,
        active: row.active,
        password: "",
      },
    });
  };

  const handleAddNew = () => {
    setNotice(null);
    setEditingRow({
      id: null,
      data: {
        username: "",
        email: "",
        role: "user",
        active: true,
        password: "",
      },
    });
  };

  const handleSave = async () => {
    if (!editingRow) return;

    const username = editingRow.data.username.trim();
    const email = editingRow.data.email.trim();
    if (!username || !email) {
      setNotice("Username and email are required.");
      return;
    }

    try {
      if (editingRow.id) {
        await updateUser.mutateAsync({
          id: editingRow.id,
          username,
          email,
          role: editingRow.data.role as User["role"],
          active: editingRow.data.active,
        });
        setNotice("User record updated through the canonical admin API.");
      } else {
        if (!editingRow.data.password.trim()) {
          setNotice("A temporary password is required for new users.");
          return;
        }
        await createUser.mutateAsync({
          username,
          name: username,
          email,
          role: editingRow.data.role as User["role"],
          active: editingRow.data.active,
          password: editingRow.data.password,
        } as Partial<User> & { username: string; active: boolean; password: string });
        setNotice("User record created through the canonical admin API.");
      }
      setEditingRow(null);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "User record save failed.");
    }
  };

  const handleDelete = async (id: string) => {
    if (!window.confirm("Delete this user through the canonical admin API?")) {
      return;
    }

    try {
      await deleteUser.mutateAsync(id);
      setNotice("User record deleted through the canonical admin API.");
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "User record delete failed.");
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold flex items-center gap-2">
            <Database className="h-6 w-6" />
            Database Editor
          </h2>
          <p className="text-gray-600 dark:text-gray-400 mt-1">
            Live table operations backed by canonical NovaCron admin contracts
          </p>
        </div>

        <Button variant="outline" size="sm" onClick={() => usersQuery.refetch()} disabled={loading}>
          <RefreshCw className={cn("h-4 w-4 mr-2", loading && "animate-spin")} />
          Refresh
        </Button>
      </div>

      <Card className="border-green-200 bg-green-50 dark:border-green-800 dark:bg-green-950">
        <CardContent className="p-4">
          <div className="flex items-center gap-2 text-green-800 dark:text-green-200">
            <CheckCircle className="h-5 w-5" />
            <span className="font-medium">
              Users table is connected to the canonical authenticated admin API.
            </span>
          </div>
        </CardContent>
      </Card>

      {notice && (
        <Card className="border-blue-200 bg-blue-50 dark:border-blue-800 dark:bg-blue-950">
          <CardContent className="p-4 text-sm text-blue-800 dark:text-blue-200">
            {notice}
          </CardContent>
        </Card>
      )}

      {Boolean(usersQuery.error) && (
        <Card className="border-red-200 bg-red-50 dark:border-red-800 dark:bg-red-950">
          <CardContent className="p-4">
            <div className="flex items-center gap-2 text-red-800 dark:text-red-200">
              <AlertTriangle className="h-5 w-5" />
              <span className="font-medium">
                Failed to load users from the canonical admin API.
              </span>
            </div>
          </CardContent>
        </Card>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        <div className="lg:col-span-1">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Database Tables</CardTitle>
              <CardDescription>Only tables with live contracts are editable.</CardDescription>
            </CardHeader>
            <CardContent className="p-0">
              <div className="space-y-1">
                {liveTables.map((table) => (
                  <button
                    key={table.name}
                    onClick={() => setSelectedTable(table.name)}
                    className={cn(
                      "w-full text-left px-4 py-3 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors",
                      selectedTable === table.name && "bg-blue-50 dark:bg-blue-950 border-r-2 border-blue-500"
                    )}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="font-medium">{table.name}</span>
                      <Badge variant={table.live ? "default" : "outline"}>
                        {table.live ? "live" : "pending"}
                      </Badge>
                    </div>
                    <div className="text-sm text-gray-600 dark:text-gray-400">
                      {table.description}
                    </div>
                  </button>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="lg:col-span-3 space-y-4">
          <Card>
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between gap-4">
                <div>
                  <CardTitle className="capitalize">{selectedTable} Table</CardTitle>
                  <CardDescription>{selectedTableInfo.contract}</CardDescription>
                </div>
                <Button
                  onClick={handleAddNew}
                  disabled={!selectedTableInfo.live || Boolean(editingRow)}
                  size="sm"
                >
                  <Plus className="h-4 w-4 mr-2" />
                  Add User
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              {!selectedTableInfo.live ? (
                <div className="rounded-md border border-amber-200 bg-amber-50 p-4 text-sm text-amber-800 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-200">
                  This table is intentionally read-only until a canonical API contract exists.
                </div>
              ) : (
                <>
                  <div className="flex items-center gap-4 mb-4">
                    <div className="relative flex-1">
                      <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
                      <Input
                        placeholder="Search users..."
                        value={searchQuery}
                        onChange={(event) => setSearchQuery(event.target.value)}
                        className="pl-10"
                      />
                    </div>
                    <Badge variant="outline">{filteredRows.length} records</Badge>
                  </div>

                  <FadeIn>
                    <div className="border rounded-md">
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead>ID</TableHead>
                            <TableHead>Username</TableHead>
                            <TableHead>Email</TableHead>
                            <TableHead>Role</TableHead>
                            <TableHead>Status</TableHead>
                            <TableHead>Updated</TableHead>
                            <TableHead>Actions</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {editingRow?.id === null && (
                            <EditableUserRow
                              editingRow={editingRow}
                              loading={loading}
                              setEditingRow={setEditingRow}
                              onSave={handleSave}
                              onCancel={() => setEditingRow(null)}
                            />
                          )}

                          {filteredRows.map((row) => (
                            editingRow?.id === row.id ? (
                              <EditableUserRow
                                key={row.id}
                                editingRow={editingRow}
                                loading={loading}
                                setEditingRow={setEditingRow}
                                onSave={handleSave}
                                onCancel={() => setEditingRow(null)}
                              />
                            ) : (
                              <TableRow key={row.id}>
                                <TableCell className="font-mono text-sm">{row.id}</TableCell>
                                <TableCell>{row.username}</TableCell>
                                <TableCell>{row.email}</TableCell>
                                <TableCell>
                                  <Badge variant="outline">{row.role}</Badge>
                                </TableCell>
                                <TableCell>
                                  <Badge
                                    variant={row.active ? "default" : "secondary"}
                                    className={row.active ? "bg-green-100 text-green-800" : ""}
                                  >
                                    {row.active ? "active" : "disabled"}
                                  </Badge>
                                </TableCell>
                                <TableCell>{formatDate(row.updated_at || row.created_at)}</TableCell>
                                <TableCell>
                                  <div className="flex items-center gap-1">
                                    <Button
                                      size="sm"
                                      variant="ghost"
                                      onClick={() => handleEdit(row)}
                                      disabled={loading || Boolean(editingRow)}
                                    >
                                      <Edit className="h-3 w-3" />
                                    </Button>
                                    <Button
                                      size="sm"
                                      variant="ghost"
                                      onClick={() => handleDelete(row.id)}
                                      disabled={loading || Boolean(editingRow)}
                                    >
                                      <Trash2 className="h-3 w-3" />
                                    </Button>
                                  </div>
                                </TableCell>
                              </TableRow>
                            )
                          ))}
                        </TableBody>
                      </Table>
                    </div>
                  </FadeIn>
                </>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}

type EditableUserRowProps = {
  editingRow: EditingRow;
  loading: boolean;
  setEditingRow: Dispatch<SetStateAction<EditingRow | null>>;
  onSave: () => void;
  onCancel: () => void;
};

function EditableUserRow({
  editingRow,
  loading,
  setEditingRow,
  onSave,
  onCancel,
}: EditableUserRowProps) {
  const updateField = (field: keyof EditingRow["data"], value: string | boolean) => {
    setEditingRow((current) => current && {
      ...current,
      data: {
        ...current.data,
        [field]: value,
      },
    });
  };

  return (
    <TableRow className="bg-blue-50 dark:bg-blue-950">
      <TableCell className="font-mono text-sm">{editingRow.id || "new"}</TableCell>
      <TableCell>
        <Input
          value={editingRow.data.username}
          onChange={(event) => updateField("username", event.target.value)}
          className="h-8"
        />
      </TableCell>
      <TableCell>
        <Input
          type="email"
          value={editingRow.data.email}
          onChange={(event) => updateField("email", event.target.value)}
          className="h-8"
        />
      </TableCell>
      <TableCell>
        <Select value={editingRow.data.role} onValueChange={(value) => updateField("role", value)}>
          <SelectTrigger className="h-8">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {roleOptions.map((role) => (
              <SelectItem key={role} value={role}>
                {role}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </TableCell>
      <TableCell>
        <Select
          value={editingRow.data.active ? "active" : "disabled"}
          onValueChange={(value) => updateField("active", value === "active")}
        >
          <SelectTrigger className="h-8">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="active">active</SelectItem>
            <SelectItem value="disabled">disabled</SelectItem>
          </SelectContent>
        </Select>
      </TableCell>
      <TableCell>
        {editingRow.id ? (
          <span className="text-sm text-gray-500">managed by API</span>
        ) : (
          <Input
            type="password"
            placeholder="Temporary password"
            value={editingRow.data.password}
            onChange={(event) => updateField("password", event.target.value)}
            className="h-8"
          />
        )}
      </TableCell>
      <TableCell>
        <div className="flex items-center gap-1">
          <Button size="sm" onClick={onSave} disabled={loading}>
            <Save className="h-3 w-3" />
          </Button>
          <Button size="sm" variant="ghost" onClick={onCancel} disabled={loading}>
            <X className="h-3 w-3" />
          </Button>
        </div>
      </TableCell>
    </TableRow>
  );
}
