package main

import (
    "log"
    "net/http"

    "github.com/aarontrelstad/inference-scheduler/internal/api"
    "github.com/aarontrelstad/inference-scheduler/internal/scheduler"
    "github.com/aarontrelstad/inference-scheduler/internal/kvcache"
)

func main() {
	cache := kvcache.New()
	scheduler := schedule.New(cache)
	handlers := api.NewHandlers(scheduler)
	router := api.NewRouter(handlers)

	log.Println("Starting scheduler on :8080")
	log.Fatal(http.ListenAndServe(":8080", router))
}
