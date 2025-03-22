#include "oatpp/web/server/HttpConnectionHandler.hpp"
#include "oatpp/network/tcp/server/ConnectionProvider.hpp"
#include "oatpp/macro/codegen.hpp"
#include "oatpp/macro/component.hpp"
#include "oatpp/json/ObjectMapper.hpp"
#include "oatpp/network/Server.hpp"
#include <iostream>
#include <filesystem>
#include <vector>

#include "models/silero_vad.h"
#include "audio.h"

// Define the DTO for response
#include OATPP_CODEGEN_BEGIN(DTO)

class SpeechTimestampDto : public oatpp::DTO {
    DTO_INIT(SpeechTimestampDto, DTO)
    DTO_FIELD(UInt32, start);
    DTO_FIELD(UInt32, end);
};

class SpeechResponseDto : public oatpp::DTO {
    DTO_INIT(SpeechResponseDto, DTO)
    DTO_FIELD(Vector<Object<SpeechTimestampDto>>, timestamps);
};

#include OATPP_CODEGEN_END(DTO)

#include OATPP_CODEGEN_BEGIN(ApiController) //<- Begin Codegen

// Define Controller
class SpeechController : public oatpp::web::server::api::ApiController {
   public:
    explicit SpeechController(const std::shared_ptr<ObjectMapper>& objectMapper)
        : oatpp::web::server::api::ApiController(objectMapper) {}

    ENDPOINT("POST", "/detect-speech", detectSpeech, BODY_STRING(String, body)) {
        try {
            // Parse the input URL from the request body
            std::string url = body->c_str();

            // Download and load the audio file
            std::filesystem::path tempDir = std::filesystem::temp_directory_path();
            std::filesystem::path fileName = speech::utils::downloadFile(url, tempDir, false, false);

            if (fileName.empty()) {
                throw std::runtime_error("Download failed or file does not exist.");
            }

            // Load the audio file
            speech::Audio audio;
            if (!audio.load(fileName)) {
                throw std::runtime_error("Failed to load audio file.");
            }

            // Initialize the SileroVAD model
            SileroVadModel vad;

            // Resample the audio to the required sample rate
            const auto audio16k = audio.to_mono().resample(vad.sample_rate);

            // Process the audio
            vad.processOnVector(audio16k.data(0));

            // Retrieve speech timestamps
            std::vector<timestamp_t> stamps = vad.get_speech_timestamps();

            // Prepare the response
            auto response = SpeechResponseDto::createShared();
            response->timestamps = {};

            for (const auto& stamp : stamps) {
                auto timestampDto = SpeechTimestampDto::createShared();
                timestampDto->start = stamp.start_s();
                timestampDto->end = stamp.end_s();
                response->timestamps->push_back(timestampDto);
            }

            return createDtoResponse(Status::CODE_200, response);
        } catch (const std::exception& e) {
            return createResponse(Status::CODE_500, e.what());
        }
    }
};
#include OATPP_CODEGEN_END(ApiController) //<- End Codegen


// Define AppComponent class for dependency injection
class AppComponent {
   public:
    // Create a Connection Provider (e.g., TCP server on port 8000)
    OATPP_CREATE_COMPONENT(std::shared_ptr<oatpp::network::ServerConnectionProvider>, serverConnectionProvider)([] {
        return oatpp::network::tcp::server::ConnectionProvider::createShared({"0.0.0.0", 8000});
    }());

    // Create a Router
    OATPP_CREATE_COMPONENT(std::shared_ptr<oatpp::web::server::HttpRouter>, httpRouter)([] {
        return oatpp::web::server::HttpRouter::createShared();
    }());

    // Create a Connection Handler
    OATPP_CREATE_COMPONENT(std::shared_ptr<oatpp::network::ConnectionHandler>, connectionHandler)([this] {
        auto router = httpRouter.getObject();
        auto handler = oatpp::web::server::HttpConnectionHandler::createShared(router);
        return handler;
    }());

    // Create ObjectMapper
    OATPP_CREATE_COMPONENT(std::shared_ptr<oatpp::data::mapping::ObjectMapper>, objectMapper)([] {
        return oatpp::json::ObjectMapper();
    }());
};

int main() {
    // Initialize Oat++ environment
    oatpp::Environment::init();

    // Create AppComponent
    AppComponent appComponent;

    // Add the controller to the router
    auto router = appComponent.httpRouter.getObject();
    router->addController(std::make_shared<SpeechController>(appComponent.objectMapper.getObject()));

    // Create the server
//    auto connectionProvider = appComponent.serverConnectionProvider.getObject();
//    auto connectionHandler = appComponent.connectionHandler.getObject();
//    oatpp::network::Server server(connectionProvider, connectionHandler);

    /* Get connection handler component */
    OATPP_COMPONENT(std::shared_ptr<oatpp::network::ConnectionHandler>, connectionHandler);

    /* Get connection provider component */
    OATPP_COMPONENT(std::shared_ptr<oatpp::network::ServerConnectionProvider>, connectionProvider);

    /* create server */
    oatpp::network::Server server(connectionProvider,
                                  connectionHandler);

    // Start the server
    OATPP_LOGd("Server", "Running on port %s", connectionProvider->getProperty("port").getData());
    server.run();

    // Destroy Oat++ environment
    oatpp::Environment::destroy();

    return 0;
}