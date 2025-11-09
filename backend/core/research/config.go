package research

import (
	"time"
)

// ResearchConfig defines configuration for research innovation system
type ResearchConfig struct {
	// Monitoring
	EnableMonitoring    bool
	ArxivCategories     []string
	MonitoringInterval  time.Duration
	KeyResearchers      []string
	Conferences         []string

	// Collaboration
	UniversityPartnerships []UniversityPartnership
	InternshipProgram      bool
	GuestLectures          bool

	// Open Source
	OpenSourceOrg      string
	ComponentsToOpen   []string
	CommunityBudget    int64

	// Innovation
	InnovationBudget   int64
	PatentTarget       int
	ResearchTimePercent float64 // 20% time

	// Prototyping
	SandboxEnabled     bool
	TimeToPrototype    time.Duration
	ABTestingEnabled   bool

	// Metrics
	TargetPapersPerYear    int
	TargetPatentsPerYear   int
	TargetStars            int
	TargetCitations        int
	TimeToProduction       time.Duration
}

// UniversityPartnership defines a university collaboration
type UniversityPartnership struct {
	Name            string
	Department      string
	Focus           []string
	KeyContacts     []string
	JointProjects   []string
	StudentCount    int
	PublicationGoal int
}

// ResearchArea defines a focus area for monitoring
type ResearchArea struct {
	Name        string
	Keywords    []string
	ArxivCats   []string
	Conferences []string
	Journals    []string
	Priority    int
}

// DefaultConfig returns production-ready configuration
func DefaultConfig() *ResearchConfig {
	return &ResearchConfig{
		EnableMonitoring: true,
		ArxivCategories: []string{
			"cs.DC", // Distributed Computing
			"cs.NI", // Networking
			"cs.AI", // Artificial Intelligence
			"cs.CR", // Cryptography
			"cs.LG", // Machine Learning
			"quant-ph", // Quantum Physics
		},
		MonitoringInterval: 24 * time.Hour,
		KeyResearchers: []string{
			"Leslie Lamport",
			"Barbara Liskov",
			"Ion Stoica",
			"Michael Jordan",
			"Yoshua Bengio",
			"Shafi Goldwasser",
			"Silvio Micali",
		},
		Conferences: []string{
			"NSDI", "OSDI", "SOSP", "SIGCOMM",
			"NeurIPS", "ICML", "ICLR",
			"IEEE S&P", "USENIX Security", "CCS",
			"QIP", "TQC",
		},
		UniversityPartnerships: []UniversityPartnership{
			{
				Name:       "MIT CSAIL",
				Department: "Computer Science",
				Focus:      []string{"distributed systems", "networking"},
				StudentCount: 5,
				PublicationGoal: 3,
			},
			{
				Name:       "Stanford AI Lab",
				Department: "Artificial Intelligence",
				Focus:      []string{"machine learning", "federated learning"},
				StudentCount: 4,
				PublicationGoal: 2,
			},
			{
				Name:       "Berkeley RISELab",
				Department: "Computer Science",
				Focus:      []string{"cloud computing", "serverless"},
				StudentCount: 3,
				PublicationGoal: 2,
			},
			{
				Name:       "CMU",
				Department: "Computer Science",
				Focus:      []string{"networking", "edge computing"},
				StudentCount: 3,
				PublicationGoal: 2,
			},
			{
				Name:       "ETH Zurich",
				Department: "Information Security",
				Focus:      []string{"cryptography", "security"},
				StudentCount: 2,
				PublicationGoal: 2,
			},
		},
		InternshipProgram: true,
		GuestLectures: true,
		OpenSourceOrg: "github.com/novacron",
		ComponentsToOpen: []string{
			"dwcp-protocol",
			"quantum-interface",
			"neuromorphic-snn",
			"blockchain-contracts",
		},
		CommunityBudget: 500000, // $500K/year
		InnovationBudget: 5000000, // $5M/year
		PatentTarget: 20,
		ResearchTimePercent: 0.20, // 20% time
		SandboxEnabled: true,
		TimeToPrototype: 14 * 24 * time.Hour, // 2 weeks
		ABTestingEnabled: true,
		TargetPapersPerYear: 10,
		TargetPatentsPerYear: 20,
		TargetStars: 10000,
		TargetCitations: 100,
		TimeToProduction: 180 * 24 * time.Hour, // 6 months
	}
}

// ResearchAreas returns predefined research focus areas
func ResearchAreas() []ResearchArea {
	return []ResearchArea{
		{
			Name: "Distributed Systems",
			Keywords: []string{
				"consensus", "distributed storage", "edge computing",
				"serverless", "microservices", "distributed transactions",
			},
			ArxivCats: []string{"cs.DC"},
			Conferences: []string{"NSDI", "OSDI", "SOSP"},
			Priority: 1,
		},
		{
			Name: "Networking",
			Keywords: []string{
				"6G", "quantum networking", "P4", "SDN",
				"NFV", "network optimization", "congestion control",
			},
			ArxivCats: []string{"cs.NI"},
			Conferences: []string{"SIGCOMM", "NSDI"},
			Priority: 1,
		},
		{
			Name: "AI/ML",
			Keywords: []string{
				"foundation models", "federated learning",
				"quantum machine learning", "neuromorphic computing",
				"privacy-preserving ML", "distributed training",
			},
			ArxivCats: []string{"cs.AI", "cs.LG"},
			Conferences: []string{"NeurIPS", "ICML", "ICLR"},
			Priority: 1,
		},
		{
			Name: "Security & Cryptography",
			Keywords: []string{
				"post-quantum cryptography", "zero-knowledge proofs",
				"homomorphic encryption", "secure MPC",
				"differential privacy", "secure enclaves",
			},
			ArxivCats: []string{"cs.CR"},
			Conferences: []string{"IEEE S&P", "USENIX Security", "CCS"},
			Priority: 1,
		},
		{
			Name: "Quantum Computing",
			Keywords: []string{
				"error correction", "quantum algorithms",
				"quantum networking", "quantum simulation",
				"NISQ", "quantum supremacy",
			},
			ArxivCats: []string{"quant-ph"},
			Conferences: []string{"QIP", "TQC"},
			Priority: 2,
		},
	}
}
