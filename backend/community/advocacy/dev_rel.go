// Package advocacy implements Developer Advocacy & Relations
// 50+ developer advocates globally, local meetups, conferences, content
// Target: 50+ advocates, 100K+ community members
package advocacy

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sync"
	"time"
)

// Advocate represents developer advocate
type Advocate struct {
	ID              string
	Name            string
	Email           string
	Region          string
	Title           string
	Expertise       []string
	Bio             string
	Photo           string
	Social          SocialProfiles
	ContentCreated  []Content
	EventsAttended  []Event
	CommunityImpact CommunityImpact
	Performance     AdvocateMetrics
	JoinedAt        time.Time
	Status          string // active, inactive
}

// SocialProfiles contains social media profiles
type SocialProfiles struct {
	Twitter     string
	LinkedIn    string
	GitHub      string
	YouTube     string
	Blog        string
	Twitch      string
	Discord     string
}

// Content represents created content
type Content struct {
	ID          string
	Type        string // blog, video, tutorial, talk, podcast
	Title       string
	Description string
	URL         string
	Duration    int // minutes for video/podcast
	Views       int
	Likes       int
	Shares      int
	Comments    int
	Engagement  float64
	PublishedAt time.Time
}

// Event represents developer event
type Event struct {
	ID              string
	Type            string // meetup, conference, workshop, webinar
	Name            string
	Description     string
	Location        string
	Virtual         bool
	Attendees       int
	Capacity        int
	Date            time.Time
	Duration        int // minutes
	Speakers        []Speaker
	Topics          []string
	Recording       string
	Slides          string
	Feedback        EventFeedback
	Cost            float64
	Sponsored       bool
	Status          string // planned, completed, cancelled
}

// Speaker represents event speaker
type Speaker struct {
	Name        string
	Title       string
	Company     string
	Topic       string
	Bio         string
	Photo       string
}

// EventFeedback contains event feedback
type EventFeedback struct {
	AverageRating   float64
	TotalResponses  int
	WouldAttendAgain float64
	Comments        []string
}

// CommunityImpact tracks advocate impact
type CommunityImpact struct {
	ReachTotal          int
	EngagementRate      float64
	CommunityGrowth     int
	DevelopersHelped    int
	IssuesResolved      int
	PRsReviewed         int
	MentoringSessions   int
	SpeakingEngagements int
}

// AdvocateMetrics tracks advocate performance
type AdvocateMetrics struct {
	ContentCreated      int
	TotalViews          int
	TotalEngagement     int
	EventsAttended      int
	EventsOrganized     int
	CommunitySize       int
	SatisfactionScore   float64
	ImpactScore         float64
	Quarter             string
}

// Meetup represents local meetup
type Meetup struct {
	ID              string
	Name            string
	Description     string
	City            string
	Country         string
	Region          string
	Organizers      []string
	Members         int
	MeetingsPerMonth int
	NextMeeting     *MeetupSession
	PastMeetings    []MeetupSession
	Topics          []string
	Venue           Venue
	Sponsors        []Sponsor
	Status          string // active, inactive
	CreatedAt       time.Time
}

// MeetupSession represents single meetup session
type MeetupSession struct {
	ID          string
	Date        time.Time
	Topic       string
	Speakers    []Speaker
	Attendees   int
	Venue       Venue
	Recording   string
	Photos      []string
	Feedback    EventFeedback
}

// Venue represents event location
type Venue struct {
	Name        string
	Address     string
	City        string
	Capacity    int
	Facilities  []string
	Virtual     bool
	StreamURL   string
}

// Sponsor represents event sponsor
type Sponsor struct {
	Name            string
	Contribution    float64
	Benefits        []string
	Logo            string
}

// Conference represents major conference
type Conference struct {
	ID              string
	Name            string
	Description     string
	Location        string
	Virtual         bool
	StartDate       time.Time
	EndDate         time.Time
	ExpectedAttendees int
	Tracks          []ConferenceTrack
	Speakers        []Speaker
	Sponsors        []Sponsor
	BoothSize       string
	BoothLocation   string
	SwagBudget      float64
	Team            []string
	Status          string
}

// ConferenceTrack represents conference track
type ConferenceTrack struct {
	Name        string
	Description string
	Sessions    []Session
	Room        string
}

// Session represents conference session
type Session struct {
	Time        time.Time
	Duration    int
	Title       string
	Speaker     string
	Description string
	Attendees   int
	Recording   string
}

// YouTubeChannel manages YouTube content
type YouTubeChannel struct {
	ChannelID       string
	ChannelName     string
	Subscribers     int
	TotalViews      int
	TotalVideos     int
	Videos          []Video
	Playlists       []Playlist
	UploadSchedule  string
	AverageViews    int
	EngagementRate  float64
}

// Video represents YouTube video
type Video struct {
	ID          string
	Title       string
	Description string
	Duration    int
	Views       int
	Likes       int
	Comments    int
	PublishedAt time.Time
	Category    string
	Tags        []string
	Thumbnail   string
	URL         string
}

// Playlist represents video playlist
type Playlist struct {
	ID          string
	Name        string
	Description string
	Videos      []string
	Views       int
	UpdatedAt   time.Time
}

// Podcast manages podcast series
type Podcast struct {
	ID              string
	Name            string
	Description     string
	Host            string
	CoHosts         []string
	Episodes        []Episode
	Subscribers     int
	TotalDownloads  int
	AverageRating   float64
	ReleaseSchedule string
	Platforms       []string
}

// Episode represents podcast episode
type Episode struct {
	Number      int
	Title       string
	Description string
	Guest       string
	Duration    int
	PublishedAt time.Time
	Downloads   int
	URL         string
	Transcript  string
	ShowNotes   string
}

// SocialMediaCampaign manages social campaigns
type SocialMediaCampaign struct {
	ID              string
	Name            string
	Description     string
	Platform        string
	StartDate       time.Time
	EndDate         time.Time
	Posts           []SocialPost
	TargetReach     int
	ActualReach     int
	Engagement      int
	Conversions     int
	Budget          float64
	Status          string
}

// SocialPost represents social media post
type SocialPost struct {
	ID          string
	Platform    string
	Content     string
	Media       []string
	Hashtags    []string
	PublishedAt time.Time
	Reach       int
	Likes       int
	Shares      int
	Comments    int
	Engagement  float64
}

// CommunityProgram represents community initiative
type CommunityProgram struct {
	ID              string
	Name            string
	Description     string
	Type            string // mentorship, champions, ambassadors
	Participants    []Participant
	Activities      []Activity
	Benefits        []string
	Requirements    []string
	ApplicationURL  string
	Status          string
	StartDate       time.Time
}

// Participant represents program participant
type Participant struct {
	UserID      string
	Name        string
	Role        string
	JoinedAt    time.Time
	Contributions int
	Status      string
}

// Activity represents program activity
type Activity struct {
	ID          string
	Type        string
	Description string
	Date        time.Time
	Participants []string
	Impact      string
}

// DevRelManager manages developer relations
type DevRelManager struct {
	mu              sync.RWMutex
	advocates       map[string]*Advocate
	meetups         map[string]*Meetup
	conferences     map[string]*Conference
	youtubeChannel  *YouTubeChannel
	podcasts        map[string]*Podcast
	campaigns       map[string]*SocialMediaCampaign
	programs        map[string]*CommunityProgram
	stats           DevRelStats
}

// DevRelStats tracks devrel metrics
type DevRelStats struct {
	TotalAdvocates      int
	ActiveAdvocates     int
	TotalMeetups        int
	MeetupsPerMonth     int
	TotalMembers        int
	ConferencesAttended int
	TotalContent        int
	ContentThisMonth    int
	TotalViews          int
	TotalEngagement     int
	CommunitySize       int
	CommunityGrowth     float64
	SocialFollowers     int
	YouTubeSubscribers  int
	PodcastSubscribers  int
	SatisfactionScore   float64
	UpdatedAt           time.Time
}

// NewDevRelManager creates devrel manager
func NewDevRelManager() *DevRelManager {
	drm := &DevRelManager{
		advocates:   make(map[string]*Advocate),
		meetups:     make(map[string]*Meetup),
		conferences: make(map[string]*Conference),
		podcasts:    make(map[string]*Podcast),
		campaigns:   make(map[string]*SocialMediaCampaign),
		programs:    make(map[string]*CommunityProgram),
		youtubeChannel: &YouTubeChannel{
			ChannelName:    "NovaCron Developers",
			Subscribers:    50000,
			TotalVideos:    500,
			Videos:         []Video{},
			Playlists:      []Playlist{},
			UploadSchedule: "weekly",
		},
	}

	drm.initializeSampleData()

	return drm
}

// initializeSampleData creates sample advocates and events
func (drm *DevRelManager) initializeSampleData() {
	// Create 50+ advocates across regions
	regions := []string{"North America", "Europe", "Asia", "Latin America", "Africa"}
	for i := 0; i < 50; i++ {
		advocate := &Advocate{
			ID:       drm.generateID("ADV"),
			Name:     fmt.Sprintf("Advocate %d", i+1),
			Email:    fmt.Sprintf("advocate%d@novacron.dev", i+1),
			Region:   regions[i%len(regions)],
			Title:    "Developer Advocate",
			Expertise: []string{"Cloud Native", "Distributed Systems", "Security"},
			Status:   "active",
			JoinedAt: time.Now().AddDate(0, -i/5, 0),
		}

		advocate.Performance = AdvocateMetrics{
			ContentCreated:  20 + i,
			TotalViews:      (i + 1) * 5000,
			EventsAttended:  10 + i/5,
			CommunitySize:   (i + 1) * 200,
			SatisfactionScore: 4.5,
			ImpactScore:     85.0,
		}

		drm.advocates[advocate.ID] = advocate
	}

	// Create 500+ meetups
	cities := []string{"San Francisco", "New York", "London", "Berlin", "Tokyo", "Singapore", "Mumbai", "São Paulo"}
	for i := 0; i < 500; i++ {
		meetup := &Meetup{
			ID:               drm.generateID("MEET"),
			Name:             fmt.Sprintf("%s NovaCron Meetup", cities[i%len(cities)]),
			City:             cities[i%len(cities)],
			Members:          50 + i,
			MeetingsPerMonth: 1,
			Status:           "active",
			CreatedAt:        time.Now().AddDate(-1, 0, 0),
		}

		drm.meetups[meetup.ID] = meetup
	}

	// Create 100+ conference events
	for i := 0; i < 100; i++ {
		conference := &Conference{
			ID:                fmt.Sprintf("CONF-%03d", i+1),
			Name:              fmt.Sprintf("Conference %d", i+1),
			ExpectedAttendees: 1000 + i*100,
			Status:            "completed",
			StartDate:         time.Now().AddDate(0, -i/10, 0),
		}

		drm.conferences[conference.ID] = conference
	}
}

// AddAdvocate adds new developer advocate
func (drm *DevRelManager) AddAdvocate(ctx context.Context, advocate *Advocate) error {
	drm.mu.Lock()
	defer drm.mu.Unlock()

	if advocate.ID == "" {
		advocate.ID = drm.generateID("ADV")
	}

	advocate.Status = "active"
	advocate.JoinedAt = time.Now()

	drm.advocates[advocate.ID] = advocate

	drm.stats.TotalAdvocates++
	drm.stats.ActiveAdvocates++
	drm.stats.UpdatedAt = time.Now()

	return nil
}

// CreateMeetup creates local meetup
func (drm *DevRelManager) CreateMeetup(ctx context.Context, meetup *Meetup) error {
	drm.mu.Lock()
	defer drm.mu.Unlock()

	if meetup.ID == "" {
		meetup.ID = drm.generateID("MEET")
	}

	meetup.Status = "active"
	meetup.CreatedAt = time.Now()

	drm.meetups[meetup.ID] = meetup

	drm.stats.TotalMeetups++
	drm.stats.TotalMembers += meetup.Members
	drm.stats.UpdatedAt = time.Now()

	return nil
}

// ScheduleConference schedules conference attendance
func (drm *DevRelManager) ScheduleConference(ctx context.Context, conference *Conference) error {
	drm.mu.Lock()
	defer drm.mu.Unlock()

	if conference.ID == "" {
		conference.ID = drm.generateID("CONF")
	}

	conference.Status = "planned"

	drm.conferences[conference.ID] = conference

	drm.stats.ConferencesAttended++
	drm.stats.UpdatedAt = time.Now()

	return nil
}

// PublishContent publishes devrel content
func (drm *DevRelManager) PublishContent(ctx context.Context, advocateID string, content *Content) error {
	drm.mu.Lock()
	defer drm.mu.Unlock()

	advocate, exists := drm.advocates[advocateID]
	if !exists {
		return fmt.Errorf("advocate not found: %s", advocateID)
	}

	if content.ID == "" {
		content.ID = drm.generateID("CONTENT")
	}

	content.PublishedAt = time.Now()

	advocate.ContentCreated = append(advocate.ContentCreated, *content)
	advocate.Performance.ContentCreated++

	drm.stats.TotalContent++
	drm.stats.ContentThisMonth++
	drm.stats.TotalViews += content.Views
	drm.stats.UpdatedAt = time.Now()

	return nil
}

// GetDevRelStats returns devrel statistics
func (drm *DevRelManager) GetDevRelStats(ctx context.Context) DevRelStats {
	drm.mu.RLock()
	defer drm.mu.RUnlock()

	stats := drm.stats

	// Count active advocates
	activeCount := 0
	for _, advocate := range drm.advocates {
		if advocate.Status == "active" {
			activeCount++
		}
	}
	stats.ActiveAdvocates = activeCount

	// Calculate meetups per month
	stats.MeetupsPerMonth = stats.TotalMeetups / 12

	// Community size
	stats.CommunitySize = stats.TotalMembers + stats.YouTubeSubscribers + stats.PodcastSubscribers

	// Social followers
	stats.SocialFollowers = stats.YouTubeSubscribers + (stats.TotalMembers * 2)

	stats.UpdatedAt = time.Now()

	return stats
}

// generateID generates unique ID
func (drm *DevRelManager) generateID(prefix string) string {
	timestamp := time.Now().UnixNano()
	hash := sha256.Sum256([]byte(fmt.Sprintf("%s-%d", prefix, timestamp)))
	return fmt.Sprintf("%s-%s", prefix, hex.EncodeToString(hash[:8]))
}

// ExportAdvocateData exports advocate data as JSON
func (drm *DevRelManager) ExportAdvocateData(ctx context.Context, advocateID string) ([]byte, error) {
	drm.mu.RLock()
	defer drm.mu.RUnlock()

	advocate, exists := drm.advocates[advocateID]
	if !exists {
		return nil, fmt.Errorf("advocate not found: %s", advocateID)
	}

	return json.MarshalIndent(advocate, "", "  ")
}
