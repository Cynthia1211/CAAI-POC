describe("CAAI-POC Backend API Tests", () => {
  it("TC-01: Backend API documentation is available", () => {
    cy.request("GET", "/openapi.json").then((response) => {
      expect(response.status).to.eq(200);
      expect(response.body).to.have.property("paths");
    });
  });

  it("TC-02: Chat API rejects a missing query", () => {
    cy.request({
      method: "POST",
      url: "/chat",
      body: {
        user_id: "TEST-001",
      },
      failOnStatusCode: false,
    }).then((response) => {
      expect(response.status).to.eq(422);
    });
  });

  it("TC-03: Voice API rejects empty text", () => {
    cy.request({
      method: "POST",
      url: "/voice/synthesize",
      body: {
        text: "",
        engine: "standard",
        voice_id: "Joanna",
      },
      failOnStatusCode: false,
    }).then((response) => {
      expect(response.status).to.eq(422);
    });
  });

  it("TC-04: Voice API rejects unsupported engines", () => {
    cy.request({
      method: "POST",
      url: "/voice/synthesize",
      body: {
        text: "Hello",
        engine: "invalid-engine",
        voice_id: "Joanna",
      },
      failOnStatusCode: false,
    }).then((response) => {
      expect(response.status).to.eq(422);
    });
  });
});
