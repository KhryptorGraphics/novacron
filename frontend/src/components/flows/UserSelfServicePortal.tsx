'use client';

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Textarea } from '@/components/ui/textarea';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { 
  Plus,
  Clock,
  CheckCircle,
  XCircle,
  AlertTriangle,
  User,
  Users,
  BarChart3,
  DollarSign,
  FileText,
  MessageSquare,
  Search,
  Filter,
  Download,
  Eye,
  Edit,
  Trash2,
  Calendar,
  TrendingUp,
  Settings,
  HelpCircle,
  BookOpen,
  Star,
  ChevronRight,
  Server,
  Database,
  Network,
  Shield
} from 'lucide-react';
import { useToast } from '@/components/ui/use-toast';

interface ResourceRequest {
  id: string;
  title: string;
  description: string;
  type: 'vm' | 'storage' | 'network' | 'database' | 'application';
  priority: 'low' | 'medium' | 'high' | 'urgent';
  status: 'draft' | 'submitted' | 'under_review' | 'approved' | 'rejected' | 'in_progress' | 'completed';
  requestedBy: string;
  costCenter: string;
  estimatedCost: number;
  requestedResources: {
    cpu?: number;
    memory?: number;
    storage?: number;
    networkBandwidth?: number;
    duration?: string;
    environment?: 'development' | 'testing' | 'staging' | 'production';
  };
  businessJustification: string;
  submittedAt: Date;
  expectedDelivery?: Date;
  approver?: string;
  approvedAt?: Date;
  comments: RequestComment[];
  attachments: string[];
}

interface RequestComment {
  id: string;
  author: string;
  role: string;
  message: string;
  timestamp: Date;
  isInternal: boolean;
}

interface UsageAnalytics {
  totalRequests: number;
  approvedRequests: number;
  pendingRequests: number;
  monthlySpend: number;
  budgetUtilization: number;
  topCategories: { type: string; count: number; spend: number }[];
  monthlyTrend: { month: string; requests: number; spend: number }[];
}

interface SupportTicket {
  id: string;
  title: string;
  category: 'technical' | 'access' | 'billing' | 'general';
  priority: 'low' | 'medium' | 'high' | 'urgent';
  status: 'open' | 'in_progress' | 'waiting_user' | 'resolved' | 'closed';
  description: string;
  createdBy: string;
  assignedTo?: string;
  createdAt: Date;
  updatedAt: Date;
  messages: TicketMessage[];
}

interface TicketMessage {
  id: string;
  author: string;
  role: 'user' | 'support' | 'admin';
  message: string;
  timestamp: Date;
  attachments?: string[];
}

interface KnowledgeArticle {
  id: string;
  title: string;
  category: string;
  content: string;
  tags: string[];
  rating: number;
  views: number;
  lastUpdated: Date;
  helpful: boolean;
}

const mockRequests: ResourceRequest[] = [
  {
    id: 'req-001',
    title: 'Development Environment for Project Alpha',
    description: 'Need development VMs for new microservices project',
    type: 'vm',
    priority: 'medium',
    status: 'approved',
    requestedBy: 'john.smith@company.com',
    costCenter: 'Engineering',
    estimatedCost: 850,
    requestedResources: {
      cpu: 8,
      memory: 16,
      storage: 500,
      environment: 'development',
      duration: '3 months'
    },
    businessJustification: 'Required for new customer portal development with expected Q2 launch',
    submittedAt: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000),
    expectedDelivery: new Date(Date.now() + 2 * 24 * 60 * 60 * 1000),
    approver: 'manager@company.com',
    approvedAt: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000),
    comments: [],
    attachments: ['architecture-diagram.pdf']
  },
  {
    id: 'req-002',
    title: 'Additional Storage for Data Analytics',
    description: 'Increase storage capacity for ML model training',
    type: 'storage',
    priority: 'high',
    status: 'under_review',
    requestedBy: 'jane.doe@company.com',
    costCenter: 'Data Science',
    estimatedCost: 1200,
    requestedResources: {
      storage: 2000,
      duration: '6 months'
    },
    businessJustification: 'Support growing dataset and improve model training performance',
    submittedAt: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000),
    comments: [
      {
        id: 'comment-1',
        author: 'approver@company.com',
        role: 'Manager',
        message: 'Can you provide more details about the expected ROI?',
        timestamp: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000),
        isInternal: false
      }
    ],
    attachments: []
  }
];

const mockAnalytics: UsageAnalytics = {
  totalRequests: 47,
  approvedRequests: 38,
  pendingRequests: 9,
  monthlySpend: 4850,
  budgetUtilization: 68.5,
  topCategories: [
    { type: 'VM', count: 22, spend: 2100 },
    { type: 'Storage', count: 15, spend: 1800 },
    { type: 'Network', count: 8, spend: 650 },
    { type: 'Database', count: 2, spend: 300 }
  ],
  monthlyTrend: Array.from({ length: 6 }, (_, i) => ({
    month: new Date(Date.now() - (5-i) * 30 * 24 * 60 * 60 * 1000).toLocaleDateString('en-US', { month: 'short' }),
    requests: Math.floor(Math.random() * 20) + 5,
    spend: Math.floor(Math.random() * 3000) + 1000
  }))
};

const mockTickets: SupportTicket[] = [
  {
    id: 'ticket-001',
    title: 'Unable to access VM console',
    category: 'technical',
    priority: 'high',
    status: 'in_progress',
    description: 'Getting timeout errors when trying to connect to VM console',
    createdBy: 'user@company.com',
    assignedTo: 'support@company.com',
    createdAt: new Date(Date.now() - 4 * 60 * 60 * 1000),
    updatedAt: new Date(Date.now() - 1 * 60 * 60 * 1000),
    messages: [
      {
        id: 'msg-1',
        author: 'support@company.com',
        role: 'support',
        message: 'We are investigating the console connectivity issue. Can you try accessing from a different network?',
        timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000)
      }
    ]
  }
];

const mockKnowledgeBase: KnowledgeArticle[] = [
  {
    id: 'kb-001',
    title: 'How to Request New Virtual Machines',
    category: 'Getting Started',
    content: 'Step-by-step guide to requesting VM resources through the self-service portal...',
    tags: ['vm', 'request', 'getting-started'],
    rating: 4.5,
    views: 342,
    lastUpdated: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000),
    helpful: true
  },
  {
    id: 'kb-002',
    title: 'Understanding Cost Center Budgets',
    category: 'Billing & Costs',
    content: 'Learn how cost center budgets work and how to track your usage...',
    tags: ['billing', 'budget', 'cost-center'],
    rating: 4.2,
    views: 187,
    lastUpdated: new Date(Date.now() - 12 * 24 * 60 * 60 * 1000),
    helpful: false
  }
];

export function UserSelfServicePortal() {
  const [activeTab, setActiveTab] = useState('overview');
  const [isCreatingRequest, setIsCreatingRequest] = useState(false);
  const [newRequest, setNewRequest] = useState<Partial<ResourceRequest>>({
    title: '',
    description: '',
    type: 'vm',
    priority: 'medium',
    costCenter: '',
    businessJustification: '',
    requestedResources: {}
  });
  const [searchTerm, setSearchTerm] = useState('');
  const [filterStatus, setFilterStatus] = useState('all');
  const [filterType, setFilterType] = useState('all');

  const { toast } = useToast();

  const submitRequest = () => {
    if (!newRequest.title || !newRequest.description || !newRequest.businessJustification) {
      toast({
        title: "Missing Information",
        description: "Please fill in all required fields.",
        variant: "destructive"
      });
      return;
    }

    toast({
      title: "Request Submitted",
      description: "Your resource request has been submitted for approval.",
    });

    setIsCreatingRequest(false);
    setNewRequest({
      title: '',
      description: '',
      type: 'vm',
      priority: 'medium',
      costCenter: '',
      businessJustification: '',
      requestedResources: {}
    });
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'approved': return 'bg-green-50 text-green-700 border-green-200';
      case 'rejected': return 'bg-red-50 text-red-700 border-red-200';
      case 'under_review': return 'bg-yellow-50 text-yellow-700 border-yellow-200';
      case 'in_progress': return 'bg-blue-50 text-blue-700 border-blue-200';
      case 'completed': return 'bg-green-50 text-green-700 border-green-200';
      case 'submitted': return 'bg-gray-50 text-gray-700 border-gray-200';
      default: return 'bg-gray-50 text-gray-700 border-gray-200';
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'urgent': return 'destructive';
      case 'high': return 'destructive';
      case 'medium': return 'default';
      case 'low': return 'secondary';
      default: return 'outline';
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'vm': return Server;
      case 'storage': return Database;
      case 'network': return Network;
      case 'database': return Database;
      default: return Settings;
    }
  };

  const filteredRequests = mockRequests.filter(request => {
    const matchesSearch = request.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         request.description.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesStatus = filterStatus === 'all' || request.status === filterStatus;
    const matchesType = filterType === 'all' || request.type === filterType;
    
    return matchesSearch && matchesStatus && matchesType;
  });

  return (
    <div className="max-w-7xl mx-auto p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold">Self-Service Portal</h2>
          <p className="text-gray-600 mt-1">Request resources, track approvals, and manage your IT services</p>
        </div>
        <Button onClick={() => setIsCreatingRequest(true)}>
          <Plus className="mr-2 h-4 w-4" />
          New Request
        </Button>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-6">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="requests">My Requests</TabsTrigger>
          <TabsTrigger value="analytics">Analytics</TabsTrigger>
          <TabsTrigger value="support">Support</TabsTrigger>
          <TabsTrigger value="knowledge">Knowledge Base</TabsTrigger>
          <TabsTrigger value="profile">Profile</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-6">
          {/* Quick Stats */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <Card>
              <CardContent className="pt-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-600">Total Requests</p>
                    <p className="text-3xl font-bold">{mockAnalytics.totalRequests}</p>
                  </div>
                  <FileText className="h-8 w-8 text-blue-600" />
                </div>
                <div className="mt-2 text-sm text-gray-600">
                  +3 this month
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="pt-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-600">Approved</p>
                    <p className="text-3xl font-bold text-green-600">{mockAnalytics.approvedRequests}</p>
                  </div>
                  <CheckCircle className="h-8 w-8 text-green-600" />
                </div>
                <div className="mt-2 text-sm text-gray-600">
                  {Math.round((mockAnalytics.approvedRequests / mockAnalytics.totalRequests) * 100)}% approval rate
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="pt-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-600">Pending</p>
                    <p className="text-3xl font-bold text-yellow-600">{mockAnalytics.pendingRequests}</p>
                  </div>
                  <Clock className="h-8 w-8 text-yellow-600" />
                </div>
                <div className="mt-2 text-sm text-gray-600">
                  Awaiting review
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="pt-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-600">Monthly Spend</p>
                    <p className="text-3xl font-bold">${mockAnalytics.monthlySpend}</p>
                  </div>
                  <DollarSign className="h-8 w-8 text-purple-600" />
                </div>
                <div className="mt-2">
                  <Progress value={mockAnalytics.budgetUtilization} className="h-2" />
                  <p className="text-xs text-gray-600 mt-1">{mockAnalytics.budgetUtilization}% of budget</p>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Recent Activity */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Recent Requests</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {mockRequests.slice(0, 5).map(request => (
                    <div key={request.id} className="flex items-center justify-between p-3 rounded-lg border">
                      <div className="flex items-center space-x-3">
                        <div className="p-1 rounded bg-blue-100">
                          {React.createElement(getTypeIcon(request.type), { className: "h-4 w-4 text-blue-600" })}
                        </div>
                        <div>
                          <p className="font-medium">{request.title}</p>
                          <p className="text-sm text-gray-600">
                            {request.type} • ${request.estimatedCost}
                          </p>
                        </div>
                      </div>
                      <div className="text-right">
                        <Badge className={getStatusColor(request.status)}>
                          {request.status.replace('_', ' ')}
                        </Badge>
                        <p className="text-xs text-gray-600 mt-1">
                          {request.submittedAt.toLocaleDateString()}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Quick Actions</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <Button variant="outline" className="w-full justify-start" onClick={() => setIsCreatingRequest(true)}>
                  <Plus className="mr-2 h-4 w-4" />
                  Request New VM
                </Button>
                <Button variant="outline" className="w-full justify-start">
                  <Database className="mr-2 h-4 w-4" />
                  Request Additional Storage
                </Button>
                <Button variant="outline" className="w-full justify-start">
                  <Network className="mr-2 h-4 w-4" />
                  Request Network Access
                </Button>
                <Button variant="outline" className="w-full justify-start">
                  <MessageSquare className="mr-2 h-4 w-4" />
                  Contact Support
                </Button>
                <Button variant="outline" className="w-full justify-start">
                  <BookOpen className="mr-2 h-4 w-4" />
                  Browse Knowledge Base
                </Button>
              </CardContent>
            </Card>
          </div>

          {/* Knowledge Base Highlights */}
          <Card>
            <CardHeader>
              <CardTitle>Popular Help Articles</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {mockKnowledgeBase.map(article => (
                  <div key={article.id} className="p-4 rounded-lg border hover:shadow-sm cursor-pointer">
                    <div className="flex items-start justify-between mb-2">
                      <h4 className="font-medium">{article.title}</h4>
                      <div className="flex items-center text-sm text-gray-500">
                        <Star className="h-3 w-3 mr-1 text-yellow-500" />
                        {article.rating}
                      </div>
                    </div>
                    <p className="text-sm text-gray-600 mb-2">{article.category}</p>
                    <div className="flex items-center justify-between text-xs text-gray-500">
                      <span>{article.views} views</span>
                      <span>Updated {article.lastUpdated.toLocaleDateString()}</span>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="requests" className="space-y-6">
          {/* Search and Filter */}
          <Card>
            <CardContent className="pt-6">
              <div className="flex flex-col md:flex-row gap-4">
                <div className="flex-1">
                  <div className="relative">
                    <Search className="absolute left-3 top-3 h-4 w-4 text-gray-400" />
                    <Input
                      placeholder="Search requests..."
                      value={searchTerm}
                      onChange={(e) => setSearchTerm(e.target.value)}
                      className="pl-10"
                    />
                  </div>
                </div>
                
                <Select value={filterStatus} onValueChange={setFilterStatus}>
                  <SelectTrigger className="w-[150px]">
                    <SelectValue placeholder="Status" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Status</SelectItem>
                    <SelectItem value="submitted">Submitted</SelectItem>
                    <SelectItem value="under_review">Under Review</SelectItem>
                    <SelectItem value="approved">Approved</SelectItem>
                    <SelectItem value="rejected">Rejected</SelectItem>
                    <SelectItem value="in_progress">In Progress</SelectItem>
                    <SelectItem value="completed">Completed</SelectItem>
                  </SelectContent>
                </Select>
                
                <Select value={filterType} onValueChange={setFilterType}>
                  <SelectTrigger className="w-[150px]">
                    <SelectValue placeholder="Type" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Types</SelectItem>
                    <SelectItem value="vm">Virtual Machine</SelectItem>
                    <SelectItem value="storage">Storage</SelectItem>
                    <SelectItem value="network">Network</SelectItem>
                    <SelectItem value="database">Database</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </CardContent>
          </Card>

          {/* Request List */}
          <div className="space-y-4">
            {filteredRequests.map(request => (
              <Card key={request.id}>
                <CardContent className="pt-4">
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex items-start space-x-4">
                      <div className="p-2 rounded bg-blue-100">
                        {React.createElement(getTypeIcon(request.type), { className: "h-5 w-5 text-blue-600" })}
                      </div>
                      <div>
                        <h4 className="font-semibold text-lg">{request.title}</h4>
                        <p className="text-gray-600 mt-1">{request.description}</p>
                        <div className="flex items-center space-x-4 mt-2 text-sm text-gray-500">
                          <span>Cost Center: {request.costCenter}</span>
                          <span>Estimated Cost: ${request.estimatedCost}</span>
                          <span>Submitted: {request.submittedAt.toLocaleDateString()}</span>
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Badge variant={getPriorityColor(request.priority)}>
                        {request.priority}
                      </Badge>
                      <Badge className={getStatusColor(request.status)}>
                        {request.status.replace('_', ' ')}
                      </Badge>
                    </div>
                  </div>
                  
                  {request.requestedResources && (
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4 p-3 bg-gray-50 rounded-lg">
                      {request.requestedResources.cpu && (
                        <div>
                          <span className="text-sm text-gray-600">CPU Cores:</span>
                          <span className="ml-1 font-medium">{request.requestedResources.cpu}</span>
                        </div>
                      )}
                      {request.requestedResources.memory && (
                        <div>
                          <span className="text-sm text-gray-600">Memory:</span>
                          <span className="ml-1 font-medium">{request.requestedResources.memory}GB</span>
                        </div>
                      )}
                      {request.requestedResources.storage && (
                        <div>
                          <span className="text-sm text-gray-600">Storage:</span>
                          <span className="ml-1 font-medium">{request.requestedResources.storage}GB</span>
                        </div>
                      )}
                      {request.requestedResources.duration && (
                        <div>
                          <span className="text-sm text-gray-600">Duration:</span>
                          <span className="ml-1 font-medium">{request.requestedResources.duration}</span>
                        </div>
                      )}
                    </div>
                  )}
                  
                  {request.comments.length > 0 && (
                    <div className="mb-4">
                      <h5 className="font-medium mb-2">Recent Comments</h5>
                      {request.comments.slice(-2).map(comment => (
                        <div key={comment.id} className="p-3 bg-blue-50 rounded-lg border-l-4 border-blue-500 mb-2">
                          <div className="flex items-center justify-between mb-1">
                            <span className="font-medium">{comment.author}</span>
                            <span className="text-xs text-gray-500">{comment.timestamp.toLocaleString()}</span>
                          </div>
                          <p className="text-sm">{comment.message}</p>
                        </div>
                      ))}
                    </div>
                  )}
                  
                  <div className="flex justify-end space-x-2">
                    <Button variant="outline" size="sm">
                      <Eye className="mr-2 h-4 w-4" />
                      View Details
                    </Button>
                    {request.status === 'draft' && (
                      <>
                        <Button variant="outline" size="sm">
                          <Edit className="mr-2 h-4 w-4" />
                          Edit
                        </Button>
                        <Button variant="outline" size="sm">
                          <Trash2 className="mr-2 h-4 w-4" />
                          Delete
                        </Button>
                      </>
                    )}
                    {request.status === 'under_review' && (
                      <Button variant="outline" size="sm">
                        <MessageSquare className="mr-2 h-4 w-4" />
                        Add Comment
                      </Button>
                    )}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="analytics" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2">
              <Card>
                <CardHeader>
                  <CardTitle>Request Trends</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-64 flex items-center justify-center text-gray-500">
                    <div className="text-center">
                      <BarChart3 className="mx-auto h-12 w-12 mb-4" />
                      <p>Request trend chart would be displayed here</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
            
            <div>
              <Card className="mb-6">
                <CardHeader>
                  <CardTitle>Budget Utilization</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-center">
                    <div className="text-3xl font-bold text-blue-600 mb-2">
                      {mockAnalytics.budgetUtilization}%
                    </div>
                    <Progress value={mockAnalytics.budgetUtilization} className="mb-2" />
                    <p className="text-sm text-gray-600">
                      ${mockAnalytics.monthlySpend} of $7,000 budget
                    </p>
                  </div>
                </CardContent>
              </Card>
              
              <Card>
                <CardHeader>
                  <CardTitle>Top Categories</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {mockAnalytics.topCategories.map((category, index) => (
                      <div key={index} className="flex items-center justify-between">
                        <div>
                          <span className="font-medium">{category.type}</span>
                          <p className="text-sm text-gray-600">{category.count} requests</p>
                        </div>
                        <span className="font-bold">${category.spend}</span>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        </TabsContent>

        <TabsContent value="support" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Create Support Ticket</CardTitle>
                <CardDescription>Get help with technical issues or questions</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label>Subject *</Label>
                  <Input placeholder="Brief description of your issue" />
                </div>
                
                <div className="space-y-2">
                  <Label>Category</Label>
                  <Select>
                    <SelectTrigger>
                      <SelectValue placeholder="Select category" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="technical">Technical Issue</SelectItem>
                      <SelectItem value="access">Access Request</SelectItem>
                      <SelectItem value="billing">Billing Question</SelectItem>
                      <SelectItem value="general">General Inquiry</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                
                <div className="space-y-2">
                  <Label>Priority</Label>
                  <Select>
                    <SelectTrigger>
                      <SelectValue placeholder="Select priority" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="low">Low</SelectItem>
                      <SelectItem value="medium">Medium</SelectItem>
                      <SelectItem value="high">High</SelectItem>
                      <SelectItem value="urgent">Urgent</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                
                <div className="space-y-2">
                  <Label>Description *</Label>
                  <Textarea 
                    placeholder="Detailed description of your issue or question"
                    rows={4}
                  />
                </div>
                
                <Button className="w-full">
                  <MessageSquare className="mr-2 h-4 w-4" />
                  Create Ticket
                </Button>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader>
                <CardTitle>My Support Tickets</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {mockTickets.map(ticket => (
                    <div key={ticket.id} className="p-3 rounded-lg border">
                      <div className="flex items-center justify-between mb-2">
                        <h4 className="font-medium">{ticket.title}</h4>
                        <Badge className={getStatusColor(ticket.status)}>
                          {ticket.status.replace('_', ' ')}
                        </Badge>
                      </div>
                      <div className="flex items-center justify-between text-sm text-gray-600">
                        <span>{ticket.category}</span>
                        <span>{ticket.createdAt.toLocaleDateString()}</span>
                      </div>
                      <div className="mt-2">
                        <Button variant="outline" size="sm">
                          <Eye className="mr-2 h-4 w-4" />
                          View Details
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="knowledge" className="space-y-6">
          <Card>
            <CardContent className="pt-6">
              <div className="relative mb-6">
                <Search className="absolute left-3 top-3 h-4 w-4 text-gray-400" />
                <Input
                  placeholder="Search knowledge base..."
                  className="pl-10"
                />
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {mockKnowledgeBase.map(article => (
                  <Card key={article.id} className="hover:shadow-md cursor-pointer">
                    <CardContent className="pt-4">
                      <div className="flex items-start justify-between mb-3">
                        <h4 className="font-semibold">{article.title}</h4>
                        <div className="flex items-center text-sm text-gray-500">
                          <Star className="h-3 w-3 mr-1 text-yellow-500" />
                          {article.rating}
                        </div>
                      </div>
                      
                      <Badge variant="outline" className="mb-3">{article.category}</Badge>
                      
                      <div className="flex flex-wrap gap-1 mb-3">
                        {article.tags.map(tag => (
                          <Badge key={tag} variant="secondary" className="text-xs">
                            {tag}
                          </Badge>
                        ))}
                      </div>
                      
                      <div className="flex items-center justify-between text-xs text-gray-500">
                        <span>{article.views} views</span>
                        <span>Updated {article.lastUpdated.toLocaleDateString()}</span>
                      </div>
                      
                      <div className="mt-3 pt-3 border-t flex justify-between">
                        <Button variant="outline" size="sm">
                          <Eye className="mr-2 h-4 w-4" />
                          Read
                        </Button>
                        <div className="flex items-center space-x-2">
                          <Button variant="outline" size="sm">
                            👍
                          </Button>
                          <Button variant="outline" size="sm">
                            👎
                          </Button>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="profile" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Profile Information</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label>Full Name</Label>
                  <Input defaultValue="John Smith" />
                </div>
                
                <div className="space-y-2">
                  <Label>Email</Label>
                  <Input defaultValue="john.smith@company.com" disabled />
                </div>
                
                <div className="space-y-2">
                  <Label>Department</Label>
                  <Input defaultValue="Engineering" />
                </div>
                
                <div className="space-y-2">
                  <Label>Cost Center</Label>
                  <Select defaultValue="engineering">
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="engineering">Engineering</SelectItem>
                      <SelectItem value="marketing">Marketing</SelectItem>
                      <SelectItem value="sales">Sales</SelectItem>
                      <SelectItem value="hr">Human Resources</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                
                <Button>Update Profile</Button>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader>
                <CardTitle>Notification Preferences</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <Label>Email notifications for request updates</Label>
                  <input type="checkbox" defaultChecked />
                </div>
                
                <div className="flex items-center justify-between">
                  <Label>SMS notifications for urgent items</Label>
                  <input type="checkbox" />
                </div>
                
                <div className="flex items-center justify-between">
                  <Label>Weekly usage reports</Label>
                  <input type="checkbox" defaultChecked />
                </div>
                
                <div className="flex items-center justify-between">
                  <Label>Budget threshold alerts</Label>
                  <input type="checkbox" defaultChecked />
                </div>
                
                <Button>Save Preferences</Button>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>

      {/* New Request Modal */}
      {isCreatingRequest && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <Card className="w-full max-w-2xl max-h-[90vh] overflow-y-auto">
            <CardHeader>
              <CardTitle>New Resource Request</CardTitle>
              <CardDescription>Submit a request for IT resources</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label>Request Title *</Label>
                  <Input 
                    placeholder="Brief title for your request"
                    value={newRequest.title}
                    onChange={(e) => setNewRequest(prev => ({ ...prev, title: e.target.value }))}
                  />
                </div>
                
                <div className="space-y-2">
                  <Label>Resource Type *</Label>
                  <Select 
                    value={newRequest.type}
                    onValueChange={(value: 'vm' | 'storage' | 'network' | 'database') => 
                      setNewRequest(prev => ({ ...prev, type: value }))
                    }
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="vm">Virtual Machine</SelectItem>
                      <SelectItem value="storage">Storage</SelectItem>
                      <SelectItem value="network">Network Access</SelectItem>
                      <SelectItem value="database">Database</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
              
              <div className="space-y-2">
                <Label>Description *</Label>
                <Textarea 
                  placeholder="Detailed description of what you need"
                  value={newRequest.description}
                  onChange={(e) => setNewRequest(prev => ({ ...prev, description: e.target.value }))}
                  rows={3}
                />
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label>Priority</Label>
                  <Select 
                    value={newRequest.priority}
                    onValueChange={(value: 'low' | 'medium' | 'high' | 'urgent') => 
                      setNewRequest(prev => ({ ...prev, priority: value }))
                    }
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="low">Low</SelectItem>
                      <SelectItem value="medium">Medium</SelectItem>
                      <SelectItem value="high">High</SelectItem>
                      <SelectItem value="urgent">Urgent</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                
                <div className="space-y-2">
                  <Label>Cost Center *</Label>
                  <Input 
                    placeholder="Your department/cost center"
                    value={newRequest.costCenter}
                    onChange={(e) => setNewRequest(prev => ({ ...prev, costCenter: e.target.value }))}
                  />
                </div>
              </div>
              
              {newRequest.type === 'vm' && (
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">VM Specifications</CardTitle>
                  </CardHeader>
                  <CardContent className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="space-y-2">
                      <Label>CPU Cores</Label>
                      <Input 
                        type="number" 
                        placeholder="4"
                        onChange={(e) => setNewRequest(prev => ({
                          ...prev,
                          requestedResources: {
                            ...prev.requestedResources,
                            cpu: parseInt(e.target.value) || 0
                          }
                        }))}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Memory (GB)</Label>
                      <Input 
                        type="number" 
                        placeholder="8"
                        onChange={(e) => setNewRequest(prev => ({
                          ...prev,
                          requestedResources: {
                            ...prev.requestedResources,
                            memory: parseInt(e.target.value) || 0
                          }
                        }))}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Storage (GB)</Label>
                      <Input 
                        type="number" 
                        placeholder="100"
                        onChange={(e) => setNewRequest(prev => ({
                          ...prev,
                          requestedResources: {
                            ...prev.requestedResources,
                            storage: parseInt(e.target.value) || 0
                          }
                        }))}
                      />
                    </div>
                  </CardContent>
                </Card>
              )}
              
              <div className="space-y-2">
                <Label>Business Justification *</Label>
                <Textarea 
                  placeholder="Explain the business need and expected benefits"
                  value={newRequest.businessJustification}
                  onChange={(e) => setNewRequest(prev => ({ ...prev, businessJustification: e.target.value }))}
                  rows={3}
                />
              </div>
            </CardContent>
            
            <div className="flex justify-end space-x-2 p-6 border-t">
              <Button variant="outline" onClick={() => setIsCreatingRequest(false)}>
                Cancel
              </Button>
              <Button onClick={submitRequest}>
                Submit Request
              </Button>
            </div>
          </Card>
        </div>
      )}
    </div>
  );
}