# Azure setup for TechCart (from zero)

You need three things on Azure. Total time is about 30 minutes; cost stays at a few dollars a month for demo use.

| Part | What it gives TechCart | `.env` keys it fills |
|---|---|---|
| 1. Resource group + budget | One place for everything, and a spending alarm | none |
| 2. Microsoft Foundry project + two model deployments | The assistant (LangGraph agents) and RAG embeddings | `AZURE_OPENAI_*` |
| 3. Entra ID app registration | "Sign in with Microsoft" for customers and admins | `ENTRA_*`, `ADMIN_EMAILS` |

LangSmith is not on Azure; sign in at smith.langchain.com for its API key.

---

## Part 1: Resource group and budget (5 min)

1. Open **https://portal.azure.com** and sign in with the account that owns your Pay-As-You-Go subscription.
2. Search **Resource groups** → **+ Create**.
   - Subscription: your Pay-As-You-Go subscription
   - Name: `rg-techcart`
   - Region: **East US 2** (use the same region in Part 2)
   - **Review + create** → **Create**
3. Search **Cost Management** → **Budgets** → **+ Add**.
   - Scope: your subscription. Name: `techcart-demo`, Reset: Monthly, Amount: `10` (USD).
   - Alert at 50% and 90% actual cost, to your e-mail. **Create**.

> Why East US 2: `gpt-5.4-mini` and `text-embedding-3-small` are both offered there with the *Global Standard* deployment type. Sweden Central is a good second choice. If a model is greyed out in a region, pick another region for the whole project; the resource and its deployments must be in a region that supports the model.

---

## Part 2: Microsoft Foundry project and model deployments (15 min)

### 2.1 Create the project

1. Open **https://ai.azure.com** and sign in with the same account.
2. Select **Create project** (or **+ New project**).
3. Fill in, expanding **Advanced options** if the fields are hidden:
   - Project name: `techcart`
   - Foundry resource: `techcart-foundry-<yourinitials>` (must be globally unique; it becomes your endpoint hostname)
   - Subscription: Pay-As-You-Go
   - Resource group: `rg-techcart`
   - Region: **East US 2**
4. **Create**. It takes 1-3 minutes and creates two things: a *Foundry resource* (holds the models, keys and endpoint) and a *project* inside it.

### 2.2 Deploy the chat model

1. In the project, open **Models + endpoints** (or **Model catalog** → search).
2. Search **gpt-5.4-mini** → select it → **Deploy** (or **Use this model**).
3. Deployment settings:
   - **Deployment name: `gpt-5.4-mini`**. Keep it identical to the model name: the code uses the name to know it is a GPT-5 reasoning model and must not send `temperature`.
   - Deployment type: **Global Standard**
   - Tokens per minute: **30K-50K** is plenty for a demo
4. **Deploy**.

### 2.3 Deploy the embeddings model

1. Search **text-embedding-3-small** → **Deploy**.
2. **Deployment name: `text-embedding-3-small`**, type **Global Standard**, TPM **120K** (indexing ~2,100 product sheets sends many short requests).
3. **Deploy**.

### 2.4 Copy the endpoint and key

Either place works:

- **Foundry portal**: project **Overview** → the **Azure OpenAI** endpoint and **API key**.
- **Azure portal**: `rg-techcart` → your Foundry resource → **Resource Management → Keys and Endpoint** → **Azure OpenAI** tab → **Endpoint** and **KEY 1**.

Use the endpoint that looks like one of these:

```
https://techcart-foundry-xx.openai.azure.com/
https://techcart-foundry-xx.cognitiveservices.azure.com/
```

Do **not** use the project endpoint (`https://….services.ai.azure.com/api/projects/techcart`). That one is for the Foundry Agents SDK, not for the Azure OpenAI client this project uses.

### 2.5 Fill `.env`

No quotes needed, no spaces around `=`:

```
AZURE_OPENAI_ENDPOINT=https://techcart-foundry-xx.openai.azure.com/
AZURE_OPENAI_API_KEY=<KEY 1>
AZURE_OPENAI_API_VERSION=2025-04-01-preview
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-5.4-mini
AZURE_OPENAI_EMBED_DEPLOYMENT=text-embedding-3-small
```

---

## Part 3: Entra ID app registration for "Sign in with Microsoft" (10 min)

1. Azure portal → search **Microsoft Entra ID** → **App registrations** → **+ New registration**.
2. Fill in:
   - Name: `TechCart`
   - Supported account types: **Accounts in any organizational directory and personal Microsoft accounts** (lets anyone with Outlook/Hotmail or a work account sign in). For a client demo restricted to one company, choose *this organizational directory only*.
   - Redirect URI: platform **Web**, value `http://localhost:8000/auth/callback`
   - **Register**.
3. On the **Overview** page copy **Application (client) ID**.
4. **Certificates & secrets** → **Client secrets** → **+ New client secret** → description `local-dev`, expiry 6 months → **Add**. Copy the **Value** column now; it is shown only once (the *Secret ID* is not what you need).
5. **API permissions**: confirm **Microsoft Graph → User.Read (Delegated)** is listed. It is added by default; nothing else is needed.
6. Fill `.env`:

```
ENTRA_TENANT=common
ENTRA_CLIENT_ID=<Application (client) ID>
ENTRA_CLIENT_SECRET=<secret Value>
ENTRA_REDIRECT_URI=http://localhost:8000/auth/callback
ADMIN_EMAILS=<the e-mail you will sign in with>
```

`ENTRA_TENANT=common` matches the account type chosen in step 2. For a single-tenant registration, use the **Directory (tenant) ID** from the Overview page instead.

When you deploy somewhere other than localhost, add that URL under **Authentication → Web → Redirect URIs** (must be `https://`) and update `ENTRA_REDIRECT_URI`.

---

## Part 4: Verify, then run

```bash
make check-azure      # endpoint DNS, embeddings call, chat + tool call, Entra settings
make bootstrap        # seed → train → index (now with Azure embeddings)
make api              # http://localhost:8000
```

Every `FAIL` line from `make check-azure` says what to change. The common ones:

| Symptom | Cause | Fix |
|---|---|---|
| `Cannot resolve …` / `nodename nor servname` | Endpoint hostname wrong, placeholder left in, or network/VPN | Re-copy from Keys and Endpoint (2.4) |
| `401` | Wrong key, or key from another resource | Re-copy KEY 1 from the same resource |
| `404 DeploymentNotFound` | `.env` deployment name differs from the one you typed in Foundry | Make them identical (names are case-sensitive) |
| `Unsupported parameter: 'temperature'` | Chat deployment not named `gpt-5…` | Rename the deployment, or redeploy as `gpt-5.4-mini` |
| `429` during `make index` | Embeddings TPM too low | Raise TPM on the embeddings deployment, re-run (cached chunks are skipped) |
| Sign-in: `AADSTS50011 redirect URI mismatch` | Redirect URI in the app registration differs from `ENTRA_REDIRECT_URI` | Make them identical, including `http` vs `https` and the port |
| Sign-in works but `/admin` says not authorised | Signed-in e-mail not in `ADMIN_EMAILS` | Add it (comma-separated), restart `make api` |

## What it costs

Entra app registrations are free. Model use is pay-per-token: a full `make index` with `text-embedding-3-small` and a few hundred chat turns with `gpt-5.4-mini` typically cost well under a dollar. Check current prices on the Azure OpenAI pricing page; the budget from Part 1 warns you before anything surprising.

## Clean up after the interview

Delete the resource group `rg-techcart` (removes the Foundry resource, project and deployments), and optionally the `TechCart` app registration.
