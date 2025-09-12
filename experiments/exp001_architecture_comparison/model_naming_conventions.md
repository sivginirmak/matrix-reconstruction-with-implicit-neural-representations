# Model Naming Conventions

## K-Planes (Line Features Only)
- **K-planes(multiply)**: Uses decoder(line_feat_x * line_feat_y)
- **K-planes(add)**: Uses decoder(line_feat_x + line_feat_y)
- These models use ONLY line features without any plane features
- The operation (multiply/add) determines how line features are combined

## GA-Planes (Line + Plane Features)
- **GA-Planes(multiply+plane)**: Uses decoder(line_feat_x * line_feat_y + plane_feat)
- **GA-Planes(add+plane)**: Uses decoder(line_feat_x + line_feat_y + plane_feat)
- These models include both line features AND low-resolution plane features
- The plane features add a global context to the line-based factorization

## NeRF (Coordinate-based)
- **NeRF(nonconvex)**: Standard ReLU-based MLP with deeper networks
- **NeRF(siren)**: SIREN-based continuous representation with sine activations
- These models encode coordinates directly through MLPs without explicit factorization

## Key Differences
1. **K-planes**: Pure factorized representation using only 1D line features
2. **GA-Planes**: Hybrid approach combining line features with 2D plane features
3. **NeRF**: Implicit representation learning features from coordinates

This naming convention accurately reflects the architectural differences and helps
distinguish between models that use only line-based factorization (K-planes) versus
those that incorporate additional planar features (GA-Planes).
