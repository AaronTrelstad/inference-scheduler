package api

import (
    "github.com/go-chi/chi/v5"
    "github.com/go-chi/chi/v5/middleware"
)

func NewRouter(h *Handlers) *chi.Mux {
	router := chi.NewRouter()

	router.Use(middleware.Logger)
	router.Use(middleware.Recoverer)
	router.Use(middleware.RequestID)

	router.Post("/inference", h.SubmitInference)
	router.Get("/inference/{id}", h.GetJob)
	router.Get("/workers", h.GetWorkers)
	router.Get("/health", h.GetHealth)
	router.Get("/metrics", h.GetMetrics)

	return router
}
