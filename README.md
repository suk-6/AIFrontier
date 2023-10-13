# Image Analyze API Server

## Overview

This is a simple API server for image analyze.

## Description

This API server is implemented by Python3 and Flask.

## API Docs

### /api/image

#### Request

-   Method: POST
-   Content-Type: application/json

```json
{
	"img": "base64 encoded image"
}
```

#### Response

-   Content-Type: application/json

```json
{
	"result": {
		"requestTime": "YYYY-MM-DD_HH:MM:SS",
		"conf": 0.7909653782844543,
		"label": 0, // 0: COVID-19, 1: NORMAL, 2: PNEUMONIA, 3: TUBERCULOSIS
		"imageWidth": 560,
		"imageHeight": 589
	},
	"img": "base64 encoded bounding box image"
}
```
