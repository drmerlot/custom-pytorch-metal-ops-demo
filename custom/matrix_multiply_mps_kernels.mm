#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>


id<MTLDevice> device = MTLCreateSystemDefaultDevice();
id<MTLCommandQueue> commandQueue = [device newCommandQueue];

// Define matrix dimensions
NSUInteger rowsA = 4, columnsA = 4;
NSUInteger rowsB = 4, columnsB = 4;
NSUInteger rowsC = rowsA, columnsC = columnsB;

// Allocate memory for matrices A, B, and C (the result)
float *matrixA = (float *)malloc(rowsA * columnsA * sizeof(float));
float *matrixB = (float *)malloc(rowsB * columnsB * sizeof(float));
float *matrixC = (float *)malloc(rowsC * columnsC * sizeof(float));

// Initialize matrix A and B with data...
// Create MPSMatrix descriptors
MPSMatrixDescriptor *descriptorA = [MPSMatrixDescriptor matrixDescriptorWithDimensions:rowsA columns:columnsA rowBytes:columnsA * sizeof(float) dataType:MPSDataTypeFloat32];
MPSMatrixDescriptor *descriptorB = [MPSMatrixDescriptor matrixDescriptorWithDimensions:rowsB columns:columnsB rowBytes:columnsB * sizeof(float) dataType:MPSDataTypeFloat32];
MPSMatrixDescriptor *descriptorC = [MPSMatrixDescriptor matrixDescriptorWithDimensions:rowsC columns:columnsC rowBytes:columnsC * sizeof(float) dataType:MPSDataTypeFloat32];

// Create MPSMatrix objects
MPSMatrix *mpsMatrixA = [[MPSMatrix alloc] initWithBuffer:[device newBufferWithBytes:matrixA length:rowsA*columnsA*sizeof(float) options:MTLResourceStorageModeShared] descriptor:descriptorA];
MPSMatrix *mpsMatrixB = [[MPSMatrix alloc] initWithBuffer:[device newBufferWithBytes:matrixB length:rowsB*columnsB*sizeof(float) options:MTLResourceStorageModeShared] descriptor:descriptorB];
MPSMatrix *mpsMatrixC = [[MPSMatrix alloc] initWithBuffer:[device newBufferWithBytes:matrixC length:rowsC*columnsC*sizeof(float) options:MTLResourceStorageModeShared] descriptor:descriptorC];

// Perform matrix multiplication
MPSMatrixMultiplication *matrixMultiplication = [[MPSMatrixMultiplication alloc] initWithDevice:device
                                                                                      transposeLeft:NO
                                                                                     transposeRight:NO
                                                                                             resultRows:rowsA
                                                                                          resultColumns:columnsB
                                                                                           interiorColumns:columnsA
                                                                                                   alpha:1.0
                                                                                                    beta:0.0];
id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
[matrixMultiplication encodeToCommandBuffer:commandBuffer leftMatrix:mpsMatrixA rightMatrix:mpsMatrixB resultMatrix:mpsMatrixC];
[commandBuffer commit];
[commandBuffer waitUntilCompleted];

// get the result out of the
memcpy(matrixC, [mpsMatrixC.data contents], rowsC * columnsC * sizeof(float));


// Your result is now in matrixC
free(matrixA);
free(matrixB);
free(matrixC);

