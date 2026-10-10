FROM golang:1.26-alpine AS build

WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY cmd ./cmd
COPY internal ./internal
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /aggregator ./cmd/aggregator

FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /aggregator /aggregator

ENV MODEL_SERVICE_URL=http://model:8001
ENV REDIS_URL=redis://redis:6379/0
EXPOSE 8000
ENTRYPOINT ["/aggregator"]
