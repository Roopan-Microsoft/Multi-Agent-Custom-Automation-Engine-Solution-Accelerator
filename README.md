# Multi-Agent-Custom-Automation-Engine – Solution Accelerator

MENU: [**USER STORY**](#user-story) \| [**QUICK DEPLOY**](#quick-deploy)  \| [**SUPPORTING DOCUMENTATION**](#supporting-documentation) \|

<h2><img src="./documentation/images/readme/userStory.png" width="64">
<br/>
User story
</h2>

### Overview

Problem:
Agentic AI systems are set to transform the way businesses operate, however it can be fairly complex to build an initial MVP to demonstrate this value. 

Solution:
The Multi-Agent-Custom Automation Engine Solution Accelerator provides a ready to go application to use as the base of the MVP, or as a reference, allowing you to hit the ground running.

### Technology Note
This accelerator uses the AutoGen framework from Microsoft Research.  This is an open source project that is maintained by [Microsoft Research’s AI Frontiers Lab](https://www.microsoft.com/research/lab/ai-frontiers/). Please see this [blog post](https://devblogs.microsoft.com/autogen/microsofts-agentic-frameworks-autogen-and-semantic-kernel/) for the latest information on using the AutoGen framework in production solutions.

### Use cases / scenarios
The multi-agent approach allows users to utilize multiple AI agents simultaneously for repeatable tasks, ensuring consistency and efficiency. 
The agents collaborate with a manager on various assignments for onboarding a new employee, such as HR and tech support AI working together to set up software accounts, configure hardware, schedule onboarding meetings, register employees for benefits, and send welcome emails. Additionally, these agents can handle tasks like procurement and drafting press releases.

### Business value
Multi-agent systems represent the next wave of Generative AI use cases, offering entirely new opportunities to drive efficiencies in your business. The Multi-Agent-Custom-Automation-Engine Solution Accelerator demonstrates several key benefits:

- **Allows people to focus on what matters:** by doing the heavy lifting involved with coordinating activities across an organization, peoples’ time is freed up to focus on their specializations.
- **Enabling GenAI to scale:** by not needing to build one application after another, organizations are able to reduce the friction of adopting GenAI across their entire organization. One capability can unlock almost unlimited use cases.
- **Applicable to most industries:** these are common challenges that most organizations face, across most industries.

Whilst still an emerging area, investing in agentic use cases, digitization and developing tools will be key to ensuring you are able to leverage these new technologies and seize the GenAI moment.

### Technical key features

This application is an AI-driven orchestration system that manages a group of AI agents to accomplish tasks based on user input. It uses a FastAPI backend to handle HTTP requests, processes them through various specialized agents, and stores stateful information using Azure Cosmos DB. The system is designed to:

- Receive input tasks from users.
- Generate a detailed plan to accomplish the task using a Planner agent.
- Execute the plan by delegating steps to specialized agents (e.g., HR, Procurement, Marketing).
- Incorporate human feedback into the workflow.
- Maintain state across sessions with persistent storage.

This system is intended for developing and deploying custom AI solutions for specific customers. This code has not been tested as an end-to-end, reliable production application- it is a foundation to help accelerate building out multi-agent systems. You are encouraged to add your own data and functions to the agents, and then you must apply your own performance and safety evaluation testing frameworks to this system before deploying it.

\
![image](./documentation/images/readme/macae-application.png)



### Products used/licenses required

-   Azure Container Application

-   Azure OpenAI

-   Azure Cosmos DB

-   The user deploying the template must have permission to create
    resources and resource groups.

### Solution accelerator architecture
![image](./documentation/images/readme/macae-architecture.png)


<h2><img src="./documentation/images/readme/oneClickDeploy.png" width="64">
<br/>
QUICK DEPLOY
</h2>

### Prerequisites

To deploy this solution accelerator, ensure you have access to an [Azure subscription](https://azure.microsoft.com/free/) with the necessary permissions to create **resource groups and resources**. Follow the steps in  [Azure Account Set Up](./docs/AzureAccountSetUp.md) 

Check the [Azure Products by Region](https://azure.microsoft.com/en-us/explore/global-infrastructure/products-by-region/table) page and select a **region** where the following services are available:  

- Azure OpenAI Service
- Azure AI Search
- [Azure Semantic Search](./docs/AzureSemanticSearchRegion.md)

### ⚠️ Important: Check Azure OpenAI Quota Availability  

➡️ To ensure sufficient quota is available in your subscription, please follow **[Quota check instructions guide](./documentation/quota_check.md)** before you deploy the solution.

| [![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/microsoft/Multi-Agent-Custom-Automation-Engine-Solution-Accelerator) | [![Open in Dev Containers](https://img.shields.io/static/v1?style=for-the-badge&label=Dev%20Containers&message=Open&color=blue&logo=visualstudiocode)](https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/microsoft/Multi-Agent-Custom-Automation-Engine-Solution-Accelerator) | 
|---|---|


<!-- [![Deploy to Azure](https://aka.ms/deploytoazurebutton)](https://portal.azure.com/#create/Microsoft.Template/uri/https%3A%2F%2Fraw.githubusercontent.com%2Fmicrosoft%2FMulti-Agent-Custom-Automation-Engine-Solution-Accelerator%2Frefs%2Fheads%2Fmain%2Fdeploy%2Fmacae-continer-oc.json) -->

### Configurable Deployment Settings

When you start the deployment, most parameters will have **default values**, but you can update the below settings by following the steps  [here](./documentation/CustomizingAzdParameters.md):  

| **Setting** | **Description** |  **Default value** |
|------------|----------------|  ------------|
| **Environment Name** | A **3-20 character alphanumeric value** used to generate a unique ID to prefix the resources. |  macaetemplate |
| **Cosmos Location** | A **less busy** region for **CosmosDB**, useful in case of availability constraints. |  eastus2 |
| **Deployment Type** | Select from a drop-down list. |  Global Standard |
| **GPT Model** | Choose from **gpt-4o** | gpt-4o |  
| **GPT Model Deployment Capacity** | Configure capacity for **GPT models**. | 50k |


### [Optional] Quota Recommendations  
By default, the **Gpt-4o model capacity** in deployment is set to **50k tokens**, so we recommend
> **For Global Standard | GPT-4o - the capacity to at least 50k tokens for optimal performance.**

To adjust quota settings if required, follow these [steps](./documentation/AzureGPTQuotaSettings.md)  

### Deployment Options
Pick from the options below to see step-by-step instructions for: GitHub Codespaces, VS Code Dev Containers, Local Environments, and Bicep deployments.

<details>
  <summary><b>Deploy in GitHub Codespaces</b></summary>

### GitHub Codespaces

You can run this solution using GitHub Codespaces. The button will open a web-based VS Code instance in your browser:

1. Open the solution accelerator (this may take several minutes):

    [![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/microsoft/Multi-Agent-Custom-Automation-Engine-Solution-Accelerator)
2. Accept the default values on the create Codespaces page
3. Open a terminal window if it is not already open
4. Continue with the [deploying steps](#deploying)

</details>

<details>
  <summary><b>Deploy in VS Code</b></summary>

 ### VS Code Dev Containers

You can run this solution in VS Code Dev Containers, which will open the project in your local VS Code using the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers):

1. Start Docker Desktop (install it if not already installed)
2. Open the project:

    [![Open in Dev Containers](https://img.shields.io/static/v1?style=for-the-badge&label=Dev%20Containers&message=Open&color=blue&logo=visualstudiocode)](https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/microsoft/Multi-Agent-Custom-Automation-Engine-Solution-Accelerator)

3. In the VS Code window that opens, once the project files show up (this may take several minutes), open a terminal window.
4. Continue with the [deploying steps](#deploying)

</details>

<details>
  <summary><b>Deploy in your local environment</b></summary>

### Local environment

To run the solution site and API backend only locally for development and debugging purposes, See the [local deployment guide](./documentation/LocalDeployment.md).

 </details>

### Manual Azure Deployment
Manual Deployment differs from the ‘Quick Deploy’ option in that it will install an Azure Container Registry (ACR) service, and relies on the installer to build and push the necessary containers to this ACR.  This allows you to build and push your own code changes and provides a sample solution you can customize based on your requirements. 

See the [local deployment guide](./documentation/ManualAzureDeployment.md).


### Deploying

Once you've opened the project in [Codespaces](#github-codespaces) or in [Dev Containers](#vs-code-dev-containers) or [locally](#local-environment), you can deploy it to Azure following the following steps. 

To change the azd parameters from the default values, follow the steps [here](./documentation/CustomizingAzdParameters.md). 


1. Login to Azure:

    ```shell
    azd auth login
    ```

    #### To authenticate with Azure Developer CLI (`azd`), use the following command with your **Tenant ID**:

    ```sh
    azd auth login --tenant-id <tenant-id>
   ```

2. Provision and deploy all the resources:

    ```shell
    azd up
    ```

3. Provide an `azd` environment name (like "macaeapp")
4. Select a subscription from your Azure account, and select a location which has quota for all the resources. 
    * This deployment will take *7-10 minutes* to provision the resources in your account and set up the solution with sample data. 
    * If you get an error or timeout with deployment, changing the location can help, as there may be availability constraints for the resources.
5. Open the [Azure Portal](https://portal.azure.com/), go to the deployed resource group, find the App Service and get the app URL from `Default domain`.

6. You can now delete the resources by running `azd down`, if you are done trying out the application. 
<!-- 6. You can now proceed to run the [development server](#development-server) to test the app locally, or if you are done trying out the app, you can delete the resources by running `azd down`. -->

<h2>
Additional Steps
</h2>

1. **Add App Authentication**
   
    Follow steps in [App Authentication](./documentation/azure_app_service_auth_setup.md) to configure authenitcation in app service.

    Note: Authentication changes can take up to 10 minutes 

2. **Deleting Resources After a Failed Deployment**

     Follow steps in [Delete Resource Group](./documentation/DeleteResourceGroup.md) If your deployment fails and you need to clean up the resources.

### Run locally and debug

To debug the solution, you can use the Cosmos and OpenAI services you have manually deployed.  To do this, you need to ensure that your Azure identity has the required permissions on the Cosmos and OpenAI services. 

- For OpenAI service, you can add yourself to the ‘Cognitive Services OpenAI User’ permission in the Access Control (IAM) pane of the Azure portal.  
- Cosmos is a little more difficult as it requires permissions be added through script.  See these examples for more information: 
  - [Use data plane role-based access control - Azure Cosmos DB for NoSQL | Microsoft Learn](https://learn.microsoft.com/en-us/azure/cosmos-db/nosql/security/how-to-grant-data-plane-role-based-access?tabs=built-in-definition%2Cpython&pivots=azure-interface-cli) 
  - [az cosmosdb sql role assignment | Microsoft Learn](https://learn.microsoft.com/en-us/cli/azure/cosmosdb/sql/role/assignment?view=azure-cli-latest#az-cosmosdb-sql-role-assignment-create) 

Add the appropriate endpoints from Cosmos and OpenAI services to your .env file.  
Note that you can configure the name of the Cosmos database in the configuration.  This can be helpful if you wish to separate the data messages generated in local debugging from those associated with the cloud based solution.  If you choose to use a different database, you will need to create that database in the Cosmos instance as this is not done automatically.

If you are using VSCode, you can use the debug configuration shown in the [local deployment guide](./documentation/LocalDeployment.md).

## Sample Questions

To help you get started, here are some [Sample Questions](./documentation/SampleQuestions.md) you can follow once your application is up and running.

<h2>
</br>
Responsible AI Transparency FAQ 
</h2>

Please refer to [Transparency FAQ](./documentation/TRANSPARENCY_FAQ.md) for responsible AI transparency details of this solution accelerator.

<h2>
Supporting documentation
</h2>

### How to customize

This solution is designed to be easily customizable. You can modify the front end site, or even build your own front end and attach to the backend API. You can further customize the backend by adding your own agents with their own specific capabilities. Deeper technical information to aid in this customization can be found in this [document](./documentation/CustomizeSolution.md).

### Costs

Pricing varies per region and usage, so it isn't possible to predict exact costs for your usage.
The majority of the Azure resources used in this infrastructure are on usage-based pricing tiers.
However, Azure Container Registry has a fixed cost per registry per day.

You can try the [Azure pricing calculator](https://azure.microsoft.com/en-us/pricing/calculator) for the resources:

* Azure AI Foundry: Free tier. [Pricing](https://azure.microsoft.com/pricing/details/ai-studio/)
* Azure AI Services: S0 tier, defaults to gpt-4o. Pricing is based on token count. [Pricing](https://azure.microsoft.com/pricing/details/cognitive-services/)
* Azure Container App: Consumption tier with 0.5 CPU, 1GiB memory/storage. Pricing is based on resource allocation, and each month allows for a certain amount of free usage. [Pricing](https://azure.microsoft.com/pricing/details/container-apps/)
* Azure Container Registry: Basic tier. [Pricing](https://azure.microsoft.com/pricing/details/container-registry/)
* Azure Cosmos DB: [Pricing](https://azure.microsoft.com/en-us/pricing/details/cosmos-db/autoscale-provisioned/)


⚠️ To avoid unnecessary costs, remember to take down your app if it's no longer in use,
either by deleting the resource group in the Portal or running `azd down`.

### Security guidelines

This template uses Azure Key Vault to store all connections to communicate between resources.

This template also uses [Managed Identity](https://learn.microsoft.com/entra/identity/managed-identities-azure-resources/overview) for local development and deployment.

To ensure continued best practices in your own repository, we recommend that anyone creating solutions based on our templates ensure that the [Github secret scanning](https://docs.github.com/code-security/secret-scanning/about-secret-scanning) setting is enabled.

You may want to consider additional security measures, such as:

* Enabling Microsoft Defender for Cloud to [secure your Azure resources](https://learn.microsoft.com/azure/security-center/defender-for-cloud).
* Protecting the Azure Container Apps instance with a [firewall](https://learn.microsoft.com/azure/container-apps/waf-app-gateway) and/or [Virtual Network](https://learn.microsoft.com/azure/container-apps/networking?tabs=workload-profiles-env%2Cazure-cli).


### Additional Resources
- [Python FastAPI documentation](https://fastapi.tiangolo.com/learn/)
- [AutoGen Framework Documentation](https://microsoft.github.io/autogen/dev/user-guide/core-user-guide/index.html)
- [Azure Container App documentation](https://learn.microsoft.com/en-us/azure/azure-functions/functions-how-to-custom-container?tabs=core-tools%2Cacr%2Cazure-cli2%2Cazure-cli&pivots=container-apps)
- [Azure OpenAI Service - Documentation, quickstarts, API reference - Azure AI services | Microsoft Learn](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/use-your-data)
- [Azure Cosmos DB documentation](https://learn.microsoft.com/en-us/azure/cosmos-db/)


## Disclaimers

To the extent that the Software includes components or code used in or derived from Microsoft products or services, including without limitation Microsoft Azure Services (collectively, “Microsoft Products and Services”), you must also comply with the Product Terms applicable to such Microsoft Products and Services. You acknowledge and agree that the license governing the Software does not grant you a license or other right to use Microsoft Products and Services. Nothing in the license or this ReadMe file will serve to supersede, amend, terminate or modify any terms in the Product Terms for any Microsoft Products and Services. 

You must also comply with all domestic and international export laws and regulations that apply to the Software, which include restrictions on destinations, end users, and end use. For further information on export restrictions, visit https://aka.ms/exporting. 

You acknowledge that the Software and Microsoft Products and Services (1) are not designed, intended or made available as a medical device(s), and (2) are not designed or intended to be a substitute for professional medical advice, diagnosis, treatment, or judgment and should not be used to replace or as a substitute for professional medical advice, diagnosis, treatment, or judgment. Customer is solely responsible for displaying and/or obtaining appropriate consents, warnings, disclaimers, and acknowledgements to end users of Customer’s implementation of the Online Services. 

You acknowledge the Software is not subject to SOC 1 and SOC 2 compliance audits. No Microsoft technology, nor any of its component technologies, including the Software, is intended or made available as a substitute for the professional advice, opinion, or judgement of a certified financial services professional. Do not use the Software to replace, substitute, or provide professional financial advice or judgment.  

BY ACCESSING OR USING THE SOFTWARE, YOU ACKNOWLEDGE THAT THE SOFTWARE IS NOT DESIGNED OR INTENDED TO SUPPORT ANY USE IN WHICH A SERVICE INTERRUPTION, DEFECT, ERROR, OR OTHER FAILURE OF THE SOFTWARE COULD RESULT IN THE DEATH OR SERIOUS BODILY INJURY OF ANY PERSON OR IN PHYSICAL OR ENVIRONMENTAL DAMAGE (COLLECTIVELY, “HIGH-RISK USE”), AND THAT YOU WILL ENSURE THAT, IN THE EVENT OF ANY INTERRUPTION, DEFECT, ERROR, OR OTHER FAILURE OF THE SOFTWARE, THE SAFETY OF PEOPLE, PROPERTY, AND THE ENVIRONMENT ARE NOT REDUCED BELOW A LEVEL THAT IS REASONABLY, APPROPRIATE, AND LEGAL, WHETHER IN GENERAL OR IN A SPECIFIC INDUSTRY. BY ACCESSING THE SOFTWARE, YOU FURTHER ACKNOWLEDGE THAT YOUR HIGH-RISK USE OF THE SOFTWARE IS AT YOUR OWN RISK.  

---


