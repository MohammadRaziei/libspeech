

#include "oatpp/web/server/HttpConnectionHandler.hpp"
#include "oatpp/network/tcp/server/ConnectionProvider.hpp"
#include "oatpp/web/server/api/ApiController.hpp"
#include "oatpp/json/ObjectMapper.hpp"
#include "oatpp/macro/codegen.hpp"
#include "oatpp/macro/component.hpp"
#include "oatpp/network/Server.hpp"


std::string html_string = "<!DOCTYPE html>\n"
    "<html>\n"
    "<head>\n"
    "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
    "<link rel=\"stylesheet\" href=\"https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css\">\n"
    "<style>\n"
    "body {\n"
    "  margin: 0;\n"
    "  font-family: Arial, Helvetica, sans-serif;\n"
    "}\n"
    "\n"
    ".topnav {\n"
    "  overflow: hidden;\n"
    "  background-color: #333;\n"
    "}\n"
    "\n"
    ".topnav a {\n"
    "  float: left;\n"
    "  display: block;\n"
    "  color: #f2f2f2;\n"
    "  text-align: center;\n"
    "  padding: 14px 16px;\n"
    "  text-decoration: none;\n"
    "  font-size: 17px;\n"
    "}\n"
    "\n"
    ".topnav a:hover {\n"
    "  background-color: #ddd;\n"
    "  color: black;\n"
    "}\n"
    "\n"
    ".topnav a.active {\n"
    "  background-color: #04AA6D;\n"
    "  color: white;\n"
    "}\n"
    "\n"
    ".topnav .icon {\n"
    "  display: none;\n"
    "}\n"
    "\n"
    "@media screen and (max-width: 600px) {\n"
    "  .topnav a:not(:first-child) {display: none;}\n"
    "  .topnav a.icon {\n"
    "    float: right;\n"
    "    display: block;\n"
    "  }\n"
    "}\n"
    "\n"
    "@media screen and (max-width: 600px) {\n"
    "  .topnav.responsive {position: relative;}\n"
    "  .topnav.responsive .icon {\n"
    "    position: absolute;\n"
    "    right: 0;\n"
    "    top: 0;\n"
    "  }\n"
    "  .topnav.responsive a {\n"
    "    float: none;\n"
    "    display: block;\n"
    "    text-align: left;\n"
    "  }\n"
    "}\n"
    "</style>\n"
    "</head>\n"
    "<body>\n"
    "\n"
    "<div class=\"topnav\" id=\"myTopnav\">\n"
    "  <a href=\"#home\" class=\"active\">Home</a>\n"
    "  <a href=\"#news\">News</a>\n"
    "  <a href=\"#contact\">Contact</a>\n"
    "  <a href=\"#about\">About</a>\n"
    "  <a href=\"javascript:void(0);\" class=\"icon\" onclick=\"myFunction()\">\n"
    "    <i class=\"fa fa-bars\"></i>\n"
    "  </a>\n"
    "</div>\n"
    "\n"
    "<div style=\"padding-left:16px\">\n"
    "  <h2>Responsive Topnav Example</h2>\n"
    "  <p>Resize the browser window to see how it works.</p>\n"
    "</div>\n"
    "\n"
    "<script>\n"
    "function myFunction() {\n"
    "  var x = document.getElementById(\"myTopnav\");\n"
    "  if (x.className === \"topnav\") {\n"
    "    x.className += \" responsive\";\n"
    "  } else {\n"
    "    x.className = \"topnav\";\n"
    "  }\n"
    "}\n"
    "</script>\n"
    "\n"
    "</body>\n"
    "</html>";

#include OATPP_CODEGEN_BEGIN(ApiController) //<- Begin Codegen

// Define Controller
class HelloWorldController : public oatpp::web::server::api::ApiController {
   public:
    explicit HelloWorldController(const std::shared_ptr<ObjectMapper>& objectMapper)
        : oatpp::web::server::api::ApiController(objectMapper) {}

    ENDPOINT("GET", "/", root) {
        auto response = createResponse(Status::CODE_200, html_string);
        response->putHeader("Content-Type", "text/html");
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

    OATPP_LOGd("Server", "Running on http://localhost:{} ...", connectionProvider->getProperty("port").toString())

    server.run();

    // Destroy Oat++
    oatpp::Environment::Environment::destroy();

    return 0;
}