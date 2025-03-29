//
// Created by mohammad on 3/20/25.
//

#ifndef LIBSPEECH_CONTROLER_H
#define LIBSPEECH_CONTROLER_H

#include "oatpp/web/server/api/ApiController.hpp"
#include "oatpp/core/macro/codegen.hpp"
#include "oatpp/core/macro/component.hpp"

#include OATPP_CODEGEN_BEGIN(ApiController)

class MyController : public oatpp::web::server::api::ApiController {
   public:
    MyController(OATPP_COMPONENT(std::shared_ptr<ObjectMapper>, objectMapper))
        : oatpp::web::server::api::ApiController(objectMapper) {}

    // Endpoint for the homepage
    ENDPOINT("GET", "/", root) {
        return createResponse(Status::CODE_200, "<h1>Hello, World!</h1>");
    }

    // Example endpoint for processing audio
    ENDPOINT("POST", "/process-audio", processAudio, BODY_STRING(String, fileUrl)) {
        // Simulate audio processing
        std::string responseMessage = "Processing audio from URL: " + fileUrl;
        return createResponse(Status::CODE_200, responseMessage);
    }
};

#include OATPP_CODEGEN_END(ApiController)


#endif  // LIBSPEECH_CONTROLER_H
