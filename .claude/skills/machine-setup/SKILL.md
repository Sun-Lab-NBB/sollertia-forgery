---
name: setting-up-machine
description: >-
  Guides users through configuring a local machine to access Sun lab assets. Covers setting the working
  directory and configuring server access credentials. Use when setting up a new machine, when configuration
  errors occur, or when the user asks about connecting to Sun lab resources.
---

# Machine Setup

Guides users through the sequence of configuring a local machine for Sun lab data workflows.

---

## Prerequisites

The MCP server must be running. Start it with: `sl-configure mcp`

---

## Setup Workflow

Follow these steps in order. Each step depends on the previous one completing successfully.

### Workflow Checklist

Copy this checklist and track your progress:

```
Machine Setup Progress:
- [ ] Step 1: Check if working directory is configured
- [ ] Step 2: Set working directory (if needed)
- [ ] Step 3: Check if server credentials are configured
- [ ] Step 4: Set server credentials (if needed)
- [ ] Step 5: Verify complete setup
```

---

## Step 1: Check working directory

The working directory stores all Sun lab configuration files and local runtime data.

### MCP Tool

```python
get_working_directory_tool()
```

**Success response:**
```
Working directory: /path/to/sun_lab_data
```

**Error response (not configured):**
```
Error: Unable to resolve the path to the local Sun lab's working directory...
```

If configured, proceed to Step 3. If not configured, proceed to Step 2.

---

## Step 2: Set working directory

You MUST ask the user where they want to store Sun lab data before calling this tool.

### MCP Tool

```python
set_working_directory_tool(directory="/path/to/sun_lab_data")
```

| Parameter   | Type  | Required | Description                                     |
|-------------|-------|----------|-------------------------------------------------|
| `directory` | `str` | Yes      | Absolute path to the working directory location |

**Common locations:**

| Platform | Suggested Path                         |
|----------|----------------------------------------|
| Linux    | `~/sun_lab_data` or `/data/sun_lab`    |
| macOS    | `~/sun_lab_data`                       |
| Windows  | `C:\sun_lab_data` or `D:\sun_lab_data` |

**Success response:**
```
Working directory set to: /path/to/sun_lab_data
```

The tool creates the directory and a `configuration` subdirectory if they do not exist.

---

## Step 3: Check server credentials

Server credentials enable access to the Sun lab compute server for data transfer and processing.

### MCP Tool

```python
get_server_configuration_tool()
```

**Success response:**
```
Server: cbsuwsun.biohpc.cornell.edu | User: username | Storage: /local/storage
```

**Error responses:**

| Response                                         | Meaning                           |
|--------------------------------------------------|-----------------------------------|
| `Error: Unable to locate 'server_configuration'` | Configuration file does not exist |
| `Error: unconfigured or contains placeholder`    | Password not set in configuration |

If configured and valid, proceed to Step 5. Otherwise, proceed to Step 4.

---

## Step 4: Set server credentials

Server configuration uses a two-step process for security. The MCP tool creates a template file with a placeholder
password, and the user must manually edit the file to add their actual password.

### Step 4a: Create configuration template

You MUST ask the user for their server username before calling this tool.

```python
create_server_configuration_template_tool(
    username="their_username",
    host="cbsuwsun.biohpc.cornell.edu",
    storage_root="/local/storage",
    working_root="/local/workdir",
    shared_directory="sun_data",
)
```

| Parameter          | Type  | Required | Default                       | Description                            |
|--------------------|-------|----------|-------------------------------|----------------------------------------|
| `username`         | `str` | Yes      | (none)                        | Server authentication username         |
| `host`             | `str` | No       | `cbsuwsun.biohpc.cornell.edu` | Server hostname or IP address          |
| `storage_root`     | `str` | No       | `/local/storage`              | Path to server's slow HDD RAID volume  |
| `working_root`     | `str` | No       | `/local/workdir`              | Path to server's fast NVME RAID volume |
| `shared_directory` | `str` | No       | `sun_data`                    | Name of shared Sun lab data directory  |

**Success response:**
```
Server configuration template created at: /path/to/configuration/server_configuration.yaml
ACTION REQUIRED: Edit the file to replace 'ENTER_YOUR_PASSWORD_HERE' with your actual password.
After editing, use get_server_configuration_tool to validate the configuration.
```

### Step 4b: User edits configuration file

Instruct the user to:

1. Open the configuration file at the path shown in the response
2. Replace `ENTER_YOUR_PASSWORD_HERE` with their actual server password
3. Save the file

### Step 4c: Validate configuration

After the user confirms they have edited the file:

```python
get_server_configuration_tool()
```

If successful, the credentials are now configured. If the response still shows a placeholder error, the user has not
correctly edited the file.

---

## Step 5: Verify complete setup

Confirm both configurations are working:

```python
get_working_directory_tool()
get_server_configuration_tool()
```

Both tools should return success responses without errors.

---

## Troubleshooting

### Working directory errors

| Error                                       | Cause                            | Solution                                  |
|---------------------------------------------|----------------------------------|-------------------------------------------|
| `Unable to resolve the path`                | Working directory not configured | Call `set_working_directory_tool()`       |
| `Directory does not exist at expected path` | Configured directory was deleted | Call `set_working_directory_tool()` again |

### Server configuration errors

| Error                                     | Cause                                 | Solution                                           |
|-------------------------------------------|---------------------------------------|----------------------------------------------------|
| `Unable to locate 'server_configuration'` | Configuration file not created        | Call `create_server_configuration_template_tool()` |
| `unconfigured or contains placeholder`    | Password not set or still placeholder | User must edit file to add real password           |

---

## Security Notes

Server credentials are stored in plain text in `server_configuration.yaml` within the working directory. The MCP tool
intentionally does not accept the password as a parameter to avoid exposing it in logs or conversation history. Users
must manually edit the file to add their password.
