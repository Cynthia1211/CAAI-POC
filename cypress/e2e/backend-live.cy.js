describe("CAAI-POC Live API Tests", () => {
  it("TC-05: Returns available Amazon Polly voices", () => {
    cy.request("GET", "/voice/options").then((response) => {
      expect(response.status).to.eq(200);

      expect(response.body.engines).to.include("standard");

      expect(response.body.voices).to.be.an("array").that.is.not.empty;

      const voiceIds = response.body.voices.map((voice) => voice.id);

      expect(voiceIds).to.include("Joanna");
    });
  });

  it("TC-06: Amazon Polly generates MP3 audio", () => {
    cy.request({
      method: "POST",
      url: "/voice/synthesize",
      body: {
        text: "Hello, this is a CAAI automated test.",
        engine: "standard",
        voice_id: "Joanna",
      },
      encoding: "binary",
      timeout: 30000,
    }).then((response) => {
      // Check the HTTP response.
      expect(response.status).to.eq(200);

      // Check that Polly returned MP3 audio.
      expect(response.headers["content-type"]).to.include("audio/mpeg");

      // Check that the audio response is not empty.
      expect(response.body).to.be.a("string");
      expect(response.body.length).to.be.greaterThan(1000);
    });
  });

  it("TC-07: RAG chatbot generates an AI response", () => {
    cy.request({
      method: "POST",
      url: "/chat",
      body: {
        query: "When is my next appointment?",
        user_id: "USER-001",
      },
      timeout: 120000,
    }).then((response) => {
      // Verify that the API request succeeded.
      expect(response.status).to.eq(200);

      // Verify that the response contains an answer.
      expect(response.body).to.have.property("answer");

      // Verify that the AI answer is not empty.
      expect(response.body.answer).to.be.a("string");
      expect(response.body.answer.trim().length).to.be.greaterThan(0);
    });
  });
});
