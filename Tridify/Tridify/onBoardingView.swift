//
//  onBoardingView.swift
//  Tridify
//
//  Created by Maged Alosali on 27/04/2023.
//

import SwiftUI

struct onBoardingView: View {
    
    @Environment(\.colorScheme) private var colorScheme
    
    @State private var rotationAngle = 0.0
    @State private var isAnimating = false
    
    private var isLightMode: Bool {
        colorScheme == .light
    }
    
    var body: some View {
        NavigationView {
            GeometryReader { geo in
                VStack {
                    Spacer()
                    LogoView()
                        .frame(width: geo.size.width*0.4)
                        .frame(width: geo.size.width)
                        .rotation3DEffect(.degrees(rotationAngle), axis: (x: 1 , y: 0, z: 1))
                        .opacity(isAnimating ? 1:0)
                        .animation(.easeInOut(duration: 3), value: isAnimating)
            
                    VStack {
                        
                        Text("Tridify")
                            .font(.system(size: geo.size.width/8))
                            .fontWeight(.light)
                            .kerning(15)

                        Text("Transform 2D images into 3D models.")
                            .font(.headline)
                            .foregroundColor(isLightMode ? .lightHeadline : .darkHeadline)
                            .fontWeight(.light)
                    }
                    .opacity(isAnimating ? 1 : 0)
                    .offset(y: isAnimating ? 0 : -40)
                    .animation(.easeOut(duration: 1), value: isAnimating)
                    Spacer()
                    
                    VStack {
                        NavigationLink {
                            // go to the sign page
                            Text("Create an account")
                        } label: {
                            HStack {
                                HStack {
                                    Spacer()
                                    Text ("Create Tridify Account")
                                        .font(.title2)
                                        .fontWeight(.medium)
                                
                                    Spacer()
                                    
                                    Image(systemName: "arrowtriangle.forward.fill")
                                    Spacer()
                                }
                                .padding()
                            }
                            .foregroundColor(isLightMode ? .white: .black)
                            .background(isLightMode ? .black.opacity(0.9): .white.opacity(0.98))
                            .clipShape(RoundedRectangle(cornerRadius: 20))
                            .frame(width: geo.size.width*0.75)
                        }
                        
                        HStack {
                            Text ("Already have an account ?")
                            NavigationLink {
                                Text("Login page")
                            } label: {
                                Text("Log In")
                                    .foregroundColor(.blue)
                            }
                        }
                        .font(.headline)
                        .fontWeight(.regular)
                        .padding(.vertical)
                    }
                    .opacity(isAnimating ? 1 : 0)
                    .offset(y: isAnimating ? 0 : 40)
                    .animation(.easeOut(duration: 1), value: isAnimating)
                    
                    Spacer()
                    
                    NavigationLink {
                        // go to the main view
                        Text("main view")
                    } label: {
                        HStack {
                            Text("Skip")
                            Image(systemName: "arrow.right")
                        }
                        .foregroundColor(isLightMode ? .black.opacity(0.4): .white.opacity(0.7))
                    }
                    .scaleEffect(isAnimating ? 1 : 0)
                    .offset(y: isAnimating ? 0 : 40)
                    .animation(.easeOut(duration: 2), value: isAnimating)
                }
            }
            .onAppear(){
                isAnimating = true
                logoRotation()
            }
        }
    }
    
    func logoRotation() {
        withAnimation(.easeInOut(duration: 2)){
            rotationAngle += 360
        }
    }
}

struct onBoardingView_Previews: PreviewProvider {
    static var previews: some View {
        onBoardingView()
            .previewDisplayName("Light")
        onBoardingView()
            .preferredColorScheme(.dark)
            .previewDisplayName("Dark")
    }
}
