package http

import (
	_ "embed"
	"io"
	"net/http"
)

// openAPISpec is the contract of this API for the frontend. It is written by hand: a change of
// the routes or of the response types goes together with a change of openapi.yaml.
//
//go:embed openapi.yaml
var openAPISpec []byte

const openAPIPath = "/docs/openapi.yaml"

// docsPage renders the specification with Swagger UI. Its script and style come from a CDN,
// pinned to a version, so the service ships no third-party assets; the page needs internet access.
const docsPage = `<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Агрегатор КТРУ — API</title>
  <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5.17.14/swagger-ui.css">
</head>
<body>
  <div id="swagger-ui"></div>
  <script src="https://unpkg.com/swagger-ui-dist@5.17.14/swagger-ui-bundle.js"></script>
  <script>
    window.ui = SwaggerUIBundle({url: "` + openAPIPath + `", dom_id: "#swagger-ui", deepLinking: true});
  </script>
</body>
</html>
`

// docs serves the Swagger UI page of the API.
func docs(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	io.WriteString(w, docsPage)
}

// openAPI serves the specification the page loads; also the file to import into other tools.
func openAPI(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/yaml; charset=utf-8")
	w.Write(openAPISpec)
}
