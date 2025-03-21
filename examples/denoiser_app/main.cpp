

#include "oatpp/web/server/HttpConnectionHandler.hpp"
#include "oatpp/network/tcp/server/ConnectionProvider.hpp"
#include "oatpp/web/server/api/ApiController.hpp"
#include "oatpp/json/ObjectMapper.hpp"
#include "oatpp/macro/codegen.hpp"
#include "oatpp/macro/component.hpp"
#include "oatpp/network/Server.hpp"



#include OATPP_CODEGEN_BEGIN(ApiController) //<- Begin Codegen

// Define Controller
class HelloWorldController : public oatpp::web::server::api::ApiController {
   public:
    explicit HelloWorldController(const std::shared_ptr<ObjectMapper>& objectMapper)
        : oatpp::web::server::api::ApiController(objectMapper) {}

    ENDPOINT("GET", "/", root) {
        auto response = createResponse(Status::CODE_200, "Hello, World!");
        response->putHeader("Content-Type", "text/plain");
        return response;
    }
};

#include OATPP_CODEGEN_END(ApiController) //<- End Codegen


// Define AppComponent
class AppComponent {
   public:
    // Create Connection Handler
    OATPP_CREATE_COMPONENT(std::shared_ptr<oatpp::network::ServerConnectionProvider>, serverConnectionProvider)([] {
        return oatpp::network::tcp::server::ConnectionProvider::createShared({"0.0.0.0", 8000});
    }());

    // Create Router
    OATPP_CREATE_COMPONENT(std::shared_ptr<oatpp::web::server::HttpRouter>, httpRouter)([] {
        return oatpp::web::server::HttpRouter::createShared();
    }());

    // Create Connection Handler
    OATPP_CREATE_COMPONENT(std::shared_ptr<oatpp::network::ConnectionHandler>, connectionHandler)([this] {
        auto router = httpRouter.getObject();
        auto handler = oatpp::web::server::HttpConnectionHandler::createShared(router);
        return handler;
    }());
};

int main() {
    // Initialize Oat++
    oatpp::Environment::Environment::init();

    // Create AppComponent
    AppComponent appComponent;

    // Create Controller
    auto router = appComponent.httpRouter.getObject();
    router->addController(std::make_shared<HelloWorldController>(nullptr));

    /* Get connection handler component */
    OATPP_COMPONENT(std::shared_ptr<oatpp::network::ConnectionHandler>, connectionHandler);

    /* Get connection provider component */
    OATPP_COMPONENT(std::shared_ptr<oatpp::network::ServerConnectionProvider>, connectionProvider);

    /* create server */
    oatpp::network::Server server(connectionProvider,
                                  connectionHandler);

    OATPP_LOGd("Server", "Running on port {}...", connectionProvider->getProperty("port").toString())

        server.run();

    // Destroy Oat++
    oatpp::Environment::Environment::destroy();

    return 0;
}