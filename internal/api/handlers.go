package api

import (
	"encoding/json"
	"net/http"
	"sync"
	"time"

	"github.com/aarontrelstad/inference-scheduler/internal/models"
	"github.com/google/uuid"
)

type Handlers struct {
	jobs  map[string]*models.InferenceJob
	mutex sync.RWMutex
}

func NewHandlers() *Handlers {
	return &Handlers{
		jobs: make(map[string]*models.InferenceJob),
	}
}

func (h *Handlers) SubmitInference(w http.ResponseWriter, r *http.Request) {
	var request models.SubmitRequest
	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		http.Error(w, "invalid request body", http.StatusBadRequest)
		return
	}

	if request.Prompt == "" {
		http.Error(w, "prompt is required", http.StatusBadRequest)
		return
	}
	if request.ModelID == "" {
		request.ModelID = "gpt2-small"
	}
	if request.DeadlineMs == 0 {
		request.DeadlineMs = 5000
	}

	job := &models.InferenceJob{
		ID:        uuid.NewString(),
		Prompt:    request.Prompt,
		ModelID:   request.ModelID,
		Status:    models.StatusPending,
		Priority:  request.Priority,
		CreatedAt: time.Now(),
		Deadline:  time.Now().Add(time.Duration(request.DeadlineMs) * time.Millisecond),
	}

	h.mutex.Lock()
	h.jobs[job.ID] = job
	h.mutex.Unlock()

	response := models.SubmitResponse{
		JobID: job.ID,
		Status: string(job.Status),
		PollURL: "/infer/" + job.ID,
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusAccepted)
	json.NewEncoder(w).Encode(response)
}

func (h *Handlers) GetJob(w http.ResponseWriter, r *http.Request) {

}

func (h *Handlers) GetWorkers(w http.ResponseWriter, r *http.Request) {

}

func (h *Handlers) GetHealth(w http.ResponseWriter, r *http.Request) {

}

func (h *Handlers) GetMetrics(w http.ResponseWriter, r *http.Request) {
	
}
