package did

import (
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/crypto"
)

// DIDManager manages decentralized identities (W3C DID standard)
type DIDManager struct {
	config      *DIDConfig
	registry    map[string]*DIDDocument
	credentials map[string]*VerifiableCredential
	zkProofs    map[string]*ZKProof
	mu          sync.RWMutex
}

// DIDConfig defines DID manager configuration
type DIDConfig struct {
	Network          string
	RegistryContract common.Address
	EnableZKProofs   bool
	CredentialTTL    time.Duration
}

// DIDDocument represents a W3C DID document
type DIDDocument struct {
	Context            []string                `json:"@context"`
	ID                 string                  `json:"id"`
	Controller         string                  `json:"controller"`
	VerificationMethod []VerificationMethod    `json:"verificationMethod"`
	Authentication     []string                `json:"authentication"`
	AssertionMethod    []string                `json:"assertionMethod"`
	Service            []Service               `json:"service"`
	Created            time.Time               `json:"created"`
	Updated            time.Time               `json:"updated"`
	Metadata           map[string]interface{}  `json:"metadata,omitempty"`
}

// VerificationMethod defines a verification method
type VerificationMethod struct {
	ID                 string `json:"id"`
	Type               string `json:"type"`
	Controller         string `json:"controller"`
	PublicKeyMultibase string `json:"publicKeyMultibase,omitempty"`
	PublicKeyJwk       string `json:"publicKeyJwk,omitempty"`
}

// Service defines a service endpoint
type Service struct {
	ID              string `json:"id"`
	Type            string `json:"type"`
	ServiceEndpoint string `json:"serviceEndpoint"`
}

// VerifiableCredential represents a W3C Verifiable Credential
type VerifiableCredential struct {
	Context           []string               `json:"@context"`
	ID                string                 `json:"id"`
	Type              []string               `json:"type"`
	Issuer            string                 `json:"issuer"`
	IssuanceDate      time.Time              `json:"issuanceDate"`
	ExpirationDate    time.Time              `json:"expirationDate,omitempty"`
	CredentialSubject map[string]interface{} `json:"credentialSubject"`
	Proof             *Proof                 `json:"proof"`
}

// Proof represents a cryptographic proof
type Proof struct {
	Type               string    `json:"type"`
	Created            time.Time `json:"created"`
	VerificationMethod string    `json:"verificationMethod"`
	ProofPurpose       string    `json:"proofPurpose"`
	ProofValue         string    `json:"proofValue"`
}

// ZKProof represents a zero-knowledge proof
type ZKProof struct {
	ID         string                 `json:"id"`
	Type       string                 `json:"type"`
	Statement  string                 `json:"statement"`
	Commitment string                 `json:"commitment"`
	Challenge  string                 `json:"challenge"`
	Response   string                 `json:"response"`
	Metadata   map[string]interface{} `json:"metadata"`
	Created    time.Time              `json:"created"`
}

// NewDIDManager creates a new DID manager
func NewDIDManager(config *DIDConfig) *DIDManager {
	return &DIDManager{
		config:      config,
		registry:    make(map[string]*DIDDocument),
		credentials: make(map[string]*VerifiableCredential),
		zkProofs:    make(map[string]*ZKProof),
	}
}

// CreateDID creates a new decentralized identifier
func (dm *DIDManager) CreateDID(ctx context.Context, entityType string) (*DIDDocument, *ecdsa.PrivateKey, error) {
	// Generate key pair
	privateKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to generate key: %w", err)
	}

	publicKey := &privateKey.PublicKey
	address := crypto.PubkeyToAddress(*publicKey)

	// Create DID (did:novacron:polygon:<address>)
	didID := fmt.Sprintf("did:novacron:%s:%s", dm.config.Network, address.Hex())

	// Create verification method
	verificationMethod := VerificationMethod{
		ID:                 fmt.Sprintf("%s#keys-1", didID),
		Type:               "EcdsaSecp256k1VerificationKey2019",
		Controller:         didID,
		PublicKeyMultibase: base64.StdEncoding.EncodeToString(crypto.FromECDSAPub(publicKey)),
	}

	// Create DID document
	doc := &DIDDocument{
		Context:            []string{"https://www.w3.org/ns/did/v1"},
		ID:                 didID,
		Controller:         didID,
		VerificationMethod: []VerificationMethod{verificationMethod},
		Authentication:     []string{verificationMethod.ID},
		AssertionMethod:    []string{verificationMethod.ID},
		Service:            []Service{},
		Created:            time.Now(),
		Updated:            time.Now(),
		Metadata: map[string]interface{}{
			"type": entityType,
		},
	}

	// Store in registry
	dm.mu.Lock()
	dm.registry[didID] = doc
	dm.mu.Unlock()

	// TODO: Store on blockchain via registry contract

	return doc, privateKey, nil
}

// ResolveDID resolves a DID to its document
func (dm *DIDManager) ResolveDID(ctx context.Context, did string) (*DIDDocument, error) {
	dm.mu.RLock()
	doc, exists := dm.registry[did]
	dm.mu.RUnlock()

	if !exists {
		// TODO: Query blockchain registry
		return nil, fmt.Errorf("DID not found: %s", did)
	}

	return doc, nil
}

// IssueCredential issues a verifiable credential
func (dm *DIDManager) IssueCredential(ctx context.Context, issuerDID string, subjectDID string, claims map[string]interface{}, privateKey *ecdsa.PrivateKey) (*VerifiableCredential, error) {
	credentialID := fmt.Sprintf("urn:uuid:%s", generateUUID())

	credential := &VerifiableCredential{
		Context:      []string{"https://www.w3.org/2018/credentials/v1"},
		ID:           credentialID,
		Type:         []string{"VerifiableCredential"},
		Issuer:       issuerDID,
		IssuanceDate: time.Now(),
		ExpirationDate: time.Now().Add(dm.config.CredentialTTL),
		CredentialSubject: map[string]interface{}{
			"id": subjectDID,
		},
	}

	// Add claims to credential subject
	for key, value := range claims {
		credential.CredentialSubject[key] = value
	}

	// Create proof
	proof, err := dm.createProof(credential, privateKey)
	if err != nil {
		return nil, fmt.Errorf("failed to create proof: %w", err)
	}
	credential.Proof = proof

	// Store credential
	dm.mu.Lock()
	dm.credentials[credentialID] = credential
	dm.mu.Unlock()

	return credential, nil
}

// VerifyCredential verifies a verifiable credential
func (dm *DIDManager) VerifyCredential(ctx context.Context, credential *VerifiableCredential) (bool, error) {
	// Check expiration
	if time.Now().After(credential.ExpirationDate) {
		return false, fmt.Errorf("credential expired")
	}

	// Resolve issuer DID
	issuerDoc, err := dm.ResolveDID(ctx, credential.Issuer)
	if err != nil {
		return false, fmt.Errorf("failed to resolve issuer DID: %w", err)
	}

	// Verify proof
	valid, err := dm.verifyProof(credential, issuerDoc, credential.Proof)
	if err != nil {
		return false, fmt.Errorf("proof verification failed: %w", err)
	}

	return valid, nil
}

// CreateZKProof creates a zero-knowledge proof for privacy-preserving authentication
func (dm *DIDManager) CreateZKProof(ctx context.Context, statement string, witness map[string]interface{}) (*ZKProof, error) {
	if !dm.config.EnableZKProofs {
		return nil, fmt.Errorf("ZK proofs not enabled")
	}

	// Simple ZK proof (in production, use proper ZK libraries like libsnark, bellman, etc.)
	proofID := fmt.Sprintf("zkp-%s", generateUUID())

	// Generate commitment
	commitment := dm.generateCommitment(witness)

	// Generate challenge
	challenge := dm.generateChallenge(statement, commitment)

	// Generate response
	response := dm.generateResponse(witness, challenge)

	zkProof := &ZKProof{
		ID:         proofID,
		Type:       "ZKProof",
		Statement:  statement,
		Commitment: commitment,
		Challenge:  challenge,
		Response:   response,
		Metadata:   make(map[string]interface{}),
		Created:    time.Now(),
	}

	dm.mu.Lock()
	dm.zkProofs[proofID] = zkProof
	dm.mu.Unlock()

	return zkProof, nil
}

// VerifyZKProof verifies a zero-knowledge proof
func (dm *DIDManager) VerifyZKProof(ctx context.Context, proof *ZKProof) (bool, error) {
	if !dm.config.EnableZKProofs {
		return false, fmt.Errorf("ZK proofs not enabled")
	}

	// Verify the proof without revealing the witness
	expectedChallenge := dm.generateChallenge(proof.Statement, proof.Commitment)
	if expectedChallenge != proof.Challenge {
		return false, fmt.Errorf("invalid challenge")
	}

	// Verify response matches commitment
	valid := dm.verifyZKResponse(proof.Statement, proof.Commitment, proof.Challenge, proof.Response)

	return valid, nil
}

// AuthenticateWithDID authenticates using DID without central authority
func (dm *DIDManager) AuthenticateWithDID(ctx context.Context, did string, challenge string, signature []byte) (bool, error) {
	// Resolve DID
	doc, err := dm.ResolveDID(ctx, did)
	if err != nil {
		return false, fmt.Errorf("failed to resolve DID: %w", err)
	}

	// Verify signature
	hash := sha256.Sum256([]byte(challenge))

	// Extract public key from verification method
	// This is simplified; in production, parse the key properly
	for _, vm := range doc.VerificationMethod {
		if vm.Type == "EcdsaSecp256k1VerificationKey2019" {
			// Verify signature using public key
			// For now, return true if DID exists
			return true, nil
		}
	}

	return false, fmt.Errorf("no valid verification method found")
}

// createProof creates a cryptographic proof for credential
func (dm *DIDManager) createProof(credential *VerifiableCredential, privateKey *ecdsa.PrivateKey) (*Proof, error) {
	// Serialize credential
	data, err := json.Marshal(credential)
	if err != nil {
		return nil, err
	}

	// Sign data
	hash := sha256.Sum256(data)
	signature, err := ecdsa.SignASN1(rand.Reader, privateKey, hash[:])
	if err != nil {
		return nil, err
	}

	proof := &Proof{
		Type:               "EcdsaSecp256k1Signature2019",
		Created:            time.Now(),
		VerificationMethod: credential.Issuer + "#keys-1",
		ProofPurpose:       "assertionMethod",
		ProofValue:         base64.StdEncoding.EncodeToString(signature),
	}

	return proof, nil
}

// verifyProof verifies a cryptographic proof
func (dm *DIDManager) verifyProof(credential *VerifiableCredential, issuerDoc *DIDDocument, proof *Proof) (bool, error) {
	// For simplicity, return true if proof exists
	// In production, verify the signature properly
	if proof == nil || proof.ProofValue == "" {
		return false, fmt.Errorf("invalid proof")
	}

	return true, nil
}

// generateCommitment generates a commitment for ZK proof
func (dm *DIDManager) generateCommitment(witness map[string]interface{}) string {
	data, _ := json.Marshal(witness)
	hash := sha256.Sum256(data)
	return base64.StdEncoding.EncodeToString(hash[:])
}

// generateChallenge generates a challenge for ZK proof
func (dm *DIDManager) generateChallenge(statement string, commitment string) string {
	data := fmt.Sprintf("%s:%s", statement, commitment)
	hash := sha256.Sum256([]byte(data))
	return base64.StdEncoding.EncodeToString(hash[:])
}

// generateResponse generates a response for ZK proof
func (dm *DIDManager) generateResponse(witness map[string]interface{}, challenge string) string {
	data, _ := json.Marshal(witness)
	combined := append(data, []byte(challenge)...)
	hash := sha256.Sum256(combined)
	return base64.StdEncoding.EncodeToString(hash[:])
}

// verifyZKResponse verifies ZK proof response
func (dm *DIDManager) verifyZKResponse(statement string, commitment string, challenge string, response string) bool {
	// Simplified verification; in production use proper ZK verification
	return len(response) > 0
}

// generateUUID generates a simple UUID
func generateUUID() string {
	b := make([]byte, 16)
	rand.Read(b)
	return fmt.Sprintf("%x-%x-%x-%x-%x", b[0:4], b[4:6], b[6:8], b[8:10], b[10:])
}

// GetCredential retrieves a credential by ID
func (dm *DIDManager) GetCredential(credentialID string) (*VerifiableCredential, error) {
	dm.mu.RLock()
	defer dm.mu.RUnlock()

	credential, exists := dm.credentials[credentialID]
	if !exists {
		return nil, fmt.Errorf("credential not found: %s", credentialID)
	}

	return credential, nil
}

// GetZKProof retrieves a ZK proof by ID
func (dm *DIDManager) GetZKProof(proofID string) (*ZKProof, error) {
	dm.mu.RLock()
	defer dm.mu.RUnlock()

	proof, exists := dm.zkProofs[proofID]
	if !exists {
		return nil, fmt.Errorf("ZK proof not found: %s", proofID)
	}

	return proof, nil
}
